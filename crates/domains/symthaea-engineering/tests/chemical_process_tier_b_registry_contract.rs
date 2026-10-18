// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Contract tests for the Tier-B candidate registry only. These tests do not run
// DWSIM or Cantera and must not be interpreted as scientific validation.

use serde_json::Value;
use std::collections::BTreeSet;

const REGISTRY_JSON: &str = include_str!(
    "../../../../docs/engineering/fixtures/CHEMICAL_PROCESS_TIER_B_CANDIDATES_V1.json"
);

fn registry() -> Value {
    serde_json::from_str(REGISTRY_JSON)
        .expect("Tier-B candidate registry must be valid JSON")
}

// Analytic-only fixture accessor. It deliberately does not invoke DWSIM.
fn mixer_inlet_mass_flows_kg_s() -> Vec<f64> {
    let root = registry();
    let mixer = root["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["case_id"].as_str() == Some("CHEM-B-TB-001"))
        .expect("water mixer candidate must exist");
    ["inlet_1", "inlet_2"]
        .into_iter()
        .map(|inlet| {
            mixer["configuration"][inlet]["mass_flow"]["value"]
                .as_f64()
                .expect("mixer inlet mass flow must be numeric")
        })
        .collect()
}

#[test]
fn registry_is_versioned_and_case_ids_are_unique() {
    let root = registry();
    assert_eq!(
        root["schema_id"].as_str(),
        Some("org.luminousdynamics.symthaea.chemical-process-tier-b-candidates")
    );
    assert_eq!(root["schema_version"].as_str(), Some("1.2.0"));
    let candidates = root["candidates"]
        .as_array()
        .expect("registry candidates must be an array");
    assert_eq!(candidates.len(), 5);

    let ids = candidates
        .iter()
        .map(|candidate| {
            candidate["case_id"]
                .as_str()
                .expect("every candidate must have a case_id")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), candidates.len());
}

#[test]
fn no_unexecuted_candidate_claims_frozen_simulator_outputs() {
    let root = registry();
    for candidate in root["candidates"].as_array().unwrap() {
        let id = candidate["case_id"].as_str().unwrap();
        assert_eq!(
            candidate["execution"]["status"].as_str(),
            Some("NOT_RUN"),
            "{id}: registry should not imply external simulator execution"
        );
        assert_eq!(
            candidate["execution"]["simulator_executed"].as_bool(),
            Some(false),
            "{id}: simulator execution must be explicit"
        );
        assert_eq!(
            candidate["reference_outputs"]["exact_simulator_outputs_frozen"].as_bool(),
            Some(false),
            "{id}: tutorial values are not frozen simulator outputs"
        );
        assert!(
            candidate["execution"]["raw_output_hash"].is_null(),
            "{id}: no raw output digest may be claimed before a run"
        );
        assert!(
            candidate["reference_outputs"]["output_digest"].is_null(),
            "{id}: no golden-output digest may be claimed before a freeze"
        );
        assert!(
            candidate["reference_outputs"]["freeze_receipt_sha256"].is_null(),
            "{id}: no receipt digest may be claimed before a freeze"
        );
    }
}

#[test]
fn every_unfrozen_candidate_has_concrete_freeze_blockers() {
    let root = registry();
    for candidate in root["candidates"].as_array().unwrap() {
        let id = candidate["case_id"].as_str().unwrap();
        let blockers = candidate["freeze_blockers"]
            .as_array()
            .unwrap_or_else(|| panic!("{id}: freeze_blockers must be an array"));
        assert!(
            !blockers.is_empty(),
            "{id}: unfrozen candidate must explain what blocks freezing"
        );
        assert!(
            blockers.iter().all(|blocker| blocker.as_str().is_some_and(|s| !s.trim().is_empty())),
            "{id}: freeze blockers must be readable strings"
        );
    }
}

#[test]
fn approximate_tutorial_hints_are_labeled_non_gold() {
    let root = registry();
    for candidate in root["candidates"].as_array().unwrap() {
        if let Some(hints) = candidate.get("source_hints_not_gold") {
            for hint in hints.as_array().expect("source_hints_not_gold must be an array") {
                assert!(
                    hint["qualifier"].as_str().is_some_and(|text| !text.trim().is_empty()),
                    "{}: tutorial hint needs a qualifier",
                    candidate["case_id"].as_str().unwrap()
                );
                assert!(
                    hint["use"].as_str().is_some_and(|text| {
                        text == "sanity_hint_only" || text == "broad_sanity_bound_only"
                    }),
                    "{}: tutorial hint cannot be marked as exact gold",
                    candidate["case_id"].as_str().unwrap()
                );
            }
        }
    }
}

#[test]
fn mixer_mass_sum_is_an_analytic_invariant_not_a_simulator_result() {
    let root = registry();
    let mixer = root["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .find(|candidate| candidate["case_id"].as_str() == Some("CHEM-B-TB-001"))
        .expect("water mixer candidate must exist");
    let invariants = mixer["analytic_invariants"]
        .as_array()
        .expect("mixer analytical invariants must be present");
    let mass_sum = invariants
        .iter()
        .find(|invariant| invariant["id"].as_str() == Some("MASS-FLOW-SUM"))
        .expect("exact mass-flow sum invariant must exist");
    assert_eq!(mass_sum["expected"]["value"].as_f64(), Some(150.0));
    assert_eq!(mass_sum["expected"]["unit"].as_str(), Some("kg/s"));
    assert_eq!(mass_sum["status"].as_str(), Some("FROZEN_ANALYTIC_INVARIANT"));
    assert_eq!(
        mixer["reference_outputs"]["exact_simulator_outputs_frozen"].as_bool(),
        Some(false),
        "the analytic mass sum must not imply simulator state variables are frozen"
    );
}


#[test]
fn analytic_mixer_mass_balance_is_stream_order_invariant() {
    let flows = mixer_inlet_mass_flows_kg_s();
    let forward = flows.iter().sum::<f64>();
    let reversed = flows.iter().rev().sum::<f64>();
    assert_eq!(forward, 150.0);
    assert_eq!(reversed, forward);
}

#[test]
fn analytic_mixer_mass_balance_scales_linearly() {
    let flows = mixer_inlet_mass_flows_kg_s();
    for factor in [2.0, 10.0, 1_000.0] {
        let scaled_total = flows.iter().map(|flow| flow * factor).sum::<f64>();
        assert_eq!(scaled_total, 150.0 * factor);
    }
}

#[test]
fn analytic_mixer_mass_balance_is_unit_equivalent() {
    let flows_kg_s = mixer_inlet_mass_flows_kg_s();
    let total_kg_s = flows_kg_s.iter().sum::<f64>();
    let total_g_s = flows_kg_s.iter().map(|flow| flow * 1_000.0).sum::<f64>();
    assert_eq!(total_kg_s, 150.0);
    assert_eq!(total_g_s, 150_000.0);
    assert_eq!(total_g_s / 1_000.0, total_kg_s);
}

#[test]
fn assurance_targets_remain_hard_gates() {
    let root = registry();
    let targets = &root["acceptance_targets"];
    assert_eq!(targets["frozen_case_provenance_coverage"]["target"].as_f64(), Some(1.0));
    assert_eq!(targets["frozen_case_provenance_coverage"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["shared_dependency_declaration_coverage"]["target"].as_f64(), Some(1.0));
    assert_eq!(targets["shared_dependency_declaration_coverage"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["missing_or_skipped_results_counted_as_pass"]["target"].as_f64(), Some(0.0));
    assert_eq!(targets["missing_or_skipped_results_counted_as_pass"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["negative_control_false_accepts"]["target"].as_f64(), Some(0.0));
    assert_eq!(targets["negative_control_false_accepts"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["transitive_shared_dependency_mutation_detection"]["target"].as_f64(), Some(1.0));
    assert_eq!(targets["transitive_shared_dependency_mutation_detection"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["malformed_lineage_graph_detection"]["target"].as_f64(), Some(1.0));
    assert_eq!(targets["malformed_lineage_graph_detection"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["frozen_receipt_digest_mutation_detection"]["target"].as_f64(), Some(1.0));
    assert_eq!(targets["frozen_receipt_digest_mutation_detection"]["hard_gate"].as_bool(), Some(true));
    assert_eq!(targets["aggregate_scores_may_mask_a_failed_hard_invariant"].as_bool(), Some(false));
}
