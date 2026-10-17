// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned public-interface scenarios for the report-only agency-budget selector.
//!
//! This corpus validates deterministic software semantics only. It does not establish
//! subjective agency, clinical suitability, actuator safety, or real-world efficacy.

use serde::Deserialize;
use std::collections::HashSet;
use symthaea_core::agency_budget::{
    evaluate_agency_budget, AgencyBudgetRequestV1, AgencySelectionStatusV1,
    AGENCY_ENVELOPE_EPSILON_V1,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioCorpus {
    schema_version: u16,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    request: AgencyBudgetRequestV1,
    expected_status: AgencySelectionStatusV1,
    expected_user_only_candidate_id: String,
    expected_unconstrained_autonomy_candidate_id: String,
    expected_selected_candidate_id: String,
    expected_selected_agency_loss_index: f32,
}

fn corpus() -> ScenarioCorpus {
    let corpus: ScenarioCorpus = serde_json::from_str(include_str!(
        "fixtures/agency-budget-scenarios-v1.json"
    ))
    .expect("versioned agency-budget corpus parses");
    assert_eq!(corpus.schema_version, 1, "unsupported corpus schema");
    assert!(corpus.scenarios.len() >= 4, "agency-budget corpus unexpectedly shrank");
    let mut ids = HashSet::new();
    for scenario in &corpus.scenarios {
        assert!(!scenario.id.trim().is_empty(), "scenario ID must be non-empty");
        assert!(ids.insert(scenario.id.as_str()), "duplicate scenario ID: {}", scenario.id);
    }
    corpus
}

fn assert_scenario_matches(scenario: &Scenario) {
    let decision = evaluate_agency_budget(&scenario.request)
        .unwrap_or_else(|error| panic!("scenario {} rejected: {error:?}", scenario.id));
    assert_eq!(
        decision.status, scenario.expected_status,
        "wrong policy status for {}",
        scenario.id
    );
    assert_eq!(
        decision.user_only_candidate_id, scenario.expected_user_only_candidate_id,
        "wrong user-only reference for {}",
        scenario.id
    );
    assert_eq!(
        decision.unconstrained_autonomy_candidate_id,
        scenario.expected_unconstrained_autonomy_candidate_id,
        "wrong unconstrained autonomy reference for {}",
        scenario.id
    );
    assert_eq!(
        decision.selected_candidate_id, scenario.expected_selected_candidate_id,
        "wrong selected candidate for {}",
        scenario.id
    );
    assert!(
        (decision.selected_agency_loss_index - scenario.expected_selected_agency_loss_index).abs()
            <= AGENCY_ENVELOPE_EPSILON_V1,
        "wrong ALI for {}: expected {}, got {}",
        scenario.id,
        scenario.expected_selected_agency_loss_index,
        decision.selected_agency_loss_index
    );
    assert!(
        decision.selected_agency_loss_index
            <= decision.agency_budget + AGENCY_ENVELOPE_EPSILON_V1,
        "selected candidate exceeded the declared agency budget in {}",
        scenario.id
    );
    assert!(
        decision.candidate_assessments[decision.selected_index].within_agency_envelope,
        "selected candidate not marked inside the envelope in {}",
        scenario.id
    );
}

#[test]
fn versioned_agency_budget_corpus_matches_expected_decisions() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        assert_scenario_matches(scenario);
    }
}

#[test]
fn agency_budget_corpus_replays_deterministically() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        let expected = evaluate_agency_budget(&scenario.request)
            .expect("fixture request should be valid");
        for replay in 0..100 {
            let actual = evaluate_agency_budget(&scenario.request)
                .expect("fixture request should be valid");
            assert_eq!(
                actual, expected,
                "scenario {} changed at replay {}",
                scenario.id, replay
            );
        }
    }
}

#[test]
fn selected_candidate_maximizes_autonomy_preference_inside_the_envelope() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        let decision = evaluate_agency_budget(&scenario.request)
            .expect("fixture request should be valid");
        let mut expected_best: Option<usize> = None;
        for (index, candidate) in decision.candidate_assessments.iter().enumerate() {
            if !candidate.within_agency_envelope {
                continue;
            }
            if expected_best.is_none_or(|best| {
                candidate.autonomy_preference_probability
                    > decision.candidate_assessments[best].autonomy_preference_probability
            }) {
                expected_best = Some(index);
            }
        }
        assert_eq!(
            expected_best,
            Some(decision.selected_index),
            "non-optimal in-envelope selection for {}",
            scenario.id
        );
    }
}
