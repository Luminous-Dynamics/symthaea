// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public-interface scenarios for the report-only assistive-mobility effort policy.
//!
//! These fixtures validate software policy semantics only. They do not establish
//! physical safety, clinical suitability, physiological effort, or hardware authority.

use serde::Deserialize;
use serde_json::Value;
use symthaea_core::effort_policy::{
    evaluate_effort_request, EffortDecisionStatusV1, EffortRequestV1,
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
    request: EffortRequestV1,
    expected_status: EffortDecisionStatusV1,
    expected_findings: Vec<Value>,
}

fn corpus() -> ScenarioCorpus {
    let corpus: ScenarioCorpus = serde_json::from_str(include_str!(
        "fixtures/effort-policy-scenarios-v1.json"
    ))
    .expect("versioned effort-policy scenario corpus parses");
    assert_eq!(corpus.schema_version, 1, "unsupported corpus schema");
    corpus
}

#[test]
fn public_effort_policy_scenario_corpus_matches_expected_dispositions() {
    let cases = corpus();
    assert!(cases.scenarios.len() >= 10, "scenario corpus unexpectedly shrank");

    for scenario in &cases.scenarios {
        let actual = evaluate_effort_request(&scenario.request);
        assert_eq!(
            actual.status, scenario.expected_status,
            "unexpected disposition for scenario {}: {:?}",
            scenario.id, actual.findings
        );

        let actual_findings = serde_json::to_value(&actual.findings)
            .expect("policy findings serialize to stable JSON");
        assert_eq!(
            actual_findings,
            Value::Array(scenario.expected_findings.clone()),
            "unexpected ordered findings for scenario {}",
            scenario.id
        );
    }
}

#[test]
fn public_effort_policy_corpus_replays_deterministically() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        let expected = evaluate_effort_request(&scenario.request);
        for replay in 0..100 {
            let actual = evaluate_effort_request(&scenario.request);
            assert_eq!(
                actual, expected,
                "scenario {} differed on replay {}",
                scenario.id, replay
            );
        }
    }
}

#[test]
fn public_effort_policy_keeps_positive_and_negative_controls() {
    let cases = corpus();
    let positive = cases
        .scenarios
        .iter()
        .filter(|case| {
            case.expected_status == EffortDecisionStatusV1::AdmissibleForSimulation
        })
        .count();
    let negative = cases
        .scenarios
        .iter()
        .filter(|case| case.expected_status == EffortDecisionStatusV1::Rejected)
        .count();

    assert!(positive >= 3, "need valid mobility, participation, and training controls");
    assert!(negative >= 5, "need diverse invalid-input controls");
}
