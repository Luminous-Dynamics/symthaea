// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned public-interface replay for the report-only wearability evaluator.
//! Passing these fixtures is software evidence only, not hardware-safety evidence.

use serde::Deserialize;
use std::collections::HashSet;
use symthaea_core::wearability::{
    evaluate_wearability_request, WearabilityDispositionV1, WearabilityFindingV1,
    WearabilityInputV1, WearabilityStageV1,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioCorpus {
    schema_version: u16,
    benchmark_id: String,
    evidence_level: String,
    claim_ceiling: String,
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Scenario {
    id: String,
    profile: String,
    stage: WearabilityStageV1,
    input: WearabilityInputV1,
    expected_disposition: WearabilityDispositionV1,
    required_receipts: Vec<String>,
    positive_control: bool,
}

fn corpus() -> ScenarioCorpus {
    let corpus: ScenarioCorpus = serde_json::from_str(include_str!(
        "../../../../docs/research/open-mobility/fixtures/wearability-lifecycle-scenarios-v1.json"
    ))
    .expect("versioned wearability lifecycle corpus parses");
    assert_eq!(corpus.schema_version, 1);
    assert_eq!(corpus.benchmark_id, "OPEN-MOBILITY-WEARABILITY-001");
    assert_eq!(corpus.evidence_level, "synthetic_replay");
    assert!(
        corpus.claim_ceiling.contains("No clinical"),
        "fixture claim ceiling must remain explicit"
    );
    assert!(corpus.scenarios.len() >= 10, "scenario corpus unexpectedly shrank");

    let mut ids = HashSet::new();
    for scenario in &corpus.scenarios {
        assert!(!scenario.id.trim().is_empty(), "scenario ID cannot be empty");
        assert!(ids.insert(scenario.id.as_str()), "duplicate scenario ID");
        assert!(!scenario.required_receipts.is_empty(), "receipt contract is required");
    }
    corpus
}

#[test]
fn wearability_fixture_dispositions_match_public_evaluator() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        let decision = evaluate_wearability_request(
            &scenario.profile,
            scenario.stage,
            &scenario.input,
        );
        assert_eq!(
            decision.disposition, scenario.expected_disposition,
            "wrong disposition for {}",
            scenario.id
        );
        assert_eq!(decision.profile_id, scenario.profile);
        assert_eq!(decision.stage, scenario.stage);
    }
}

#[test]
fn wearability_replay_is_deterministic() {
    let cases = corpus();
    for scenario in &cases.scenarios {
        let expected = evaluate_wearability_request(
            &scenario.profile,
            scenario.stage,
            &scenario.input,
        );
        for replay in 0..100 {
            let actual = evaluate_wearability_request(
                &scenario.profile,
                scenario.stage,
                &scenario.input,
            );
            assert_eq!(actual, expected, "scenario {} replay {}", scenario.id, replay);
        }
    }
}

#[test]
fn known_scenarios_emit_specific_machine_readable_reasons() {
    let cases = corpus();
    let fixture_cases = [
        ("WRB-002-missing-required-fastening", WearabilityFindingV1::RequiredFasteningIncomplete),
        ("WRB-003-fit-evidence-stale-after-movement", WearabilityFindingV1::FitEvidenceStale),
        ("WRB-006-user-requests-doff", WearabilityFindingV1::UserRequestedDoff),
        ("WRB-013-task-outside-profile", WearabilityFindingV1::TaskOutsideProfile),
        ("WRB-015-thermal-discomfort-reported", WearabilityFindingV1::ThermalDiscomfortReported),
        ("WRB-016-duty-cycle-limit-reached", WearabilityFindingV1::DutyCycleLimitReached),
        ("WRB-018-hygiene-context-incompatible", WearabilityFindingV1::HygieneWorkflowIncompatible),
    ];

    for (scenario_id, expected_finding) in fixture_cases {
        let scenario = cases.scenarios.iter()
            .find(|scenario| scenario.id == scenario_id)
            .expect("expected reason-code fixture exists");
        let decision = evaluate_wearability_request(
            &scenario.profile,
            scenario.stage,
            &scenario.input,
        );
        assert!(
            decision.findings.contains(&expected_finding),
            "scenario {} missing finding {:?}; got {:?}",
            scenario_id,
            expected_finding,
            decision.findings
        );
    }

    let valid = cases.scenarios.iter()
        .find(|scenario| scenario.id == "WRB-001-verified-self-don")
        .expect("positive control exists");
    let ready = evaluate_wearability_request(&valid.profile, valid.stage, &valid.input);
    assert!(ready.findings.contains(&WearabilityFindingV1::FitCheckAdmittedForSimulation));
}

#[test]
fn findings_keep_stable_priority_order_when_multiple_conditions_coexist() {
    let input = WearabilityInputV1 {
        schema_version: 99,
        user_requested_doff: Some(true),
        fit_evidence: Some(symthaea_core::wearability::FitEvidenceV1::Stale),
        fastening_evidence: Some(symthaea_core::wearability::FasteningEvidenceV1::Incomplete),
        ..Default::default()
    };
    let decision = evaluate_wearability_request("profile-a", WearabilityStageV1::Perform, &input);
    assert_eq!(decision.disposition, WearabilityDispositionV1::DoffRequestPreemptsNewMotion);
    assert_eq!(
        decision.findings,
        vec![
            WearabilityFindingV1::UnsupportedSchemaVersion,
            WearabilityFindingV1::UserRequestedDoff,
            WearabilityFindingV1::RequiredFasteningIncomplete,
            WearabilityFindingV1::FitEvidenceStale,
        ]
    );
}

#[test]
fn corpus_preserves_positive_controls_and_blocked_workflows() {
    let cases = corpus();
    let positive = cases.scenarios.iter()
        .filter(|scenario| scenario.positive_control)
        .count();
    let ready = cases.scenarios.iter()
        .filter(|scenario| scenario.expected_disposition == WearabilityDispositionV1::ReadyForSimulation)
        .count();
    let doff = cases.scenarios.iter()
        .filter(|scenario| scenario.expected_disposition == WearabilityDispositionV1::DoffRequestPreemptsNewMotion)
        .count();
    let blocked = cases.scenarios.iter()
        .filter(|scenario| matches!(
            scenario.expected_disposition,
            WearabilityDispositionV1::StartNotAdmitted
                | WearabilityDispositionV1::FitRecheckRequired
                | WearabilityDispositionV1::InterfaceStateUnknown
                | WearabilityDispositionV1::WorkflowBlockedWithSupportPath
                | WearabilityDispositionV1::ExplicitModeSelectionRequired
                | WearabilityDispositionV1::TaskPlanRequiresReviewOrFallback
                | WearabilityDispositionV1::ReturnToServiceNotAdmitted
                | WearabilityDispositionV1::RemainUnreadyAndOfferAccessibleOptions
                | WearabilityDispositionV1::MigrationMetricMissing
                | WearabilityDispositionV1::TaskNotAdmittedForProfile
                | WearabilityDispositionV1::TaskReviewOrAlternateProfileRequired
        ))
        .count();

    let reviewed_service = cases.scenarios.iter()
        .filter(|scenario| scenario.expected_disposition == WearabilityDispositionV1::EligibleForReviewedReturnToService)
        .count();

    assert!(positive >= 3, "successful fit, removal and service controls must remain");
    assert!(ready >= 1, "corpus needs a successful fit/start control");
    assert!(reviewed_service >= 1, "corpus needs a successful service control");
    assert!(doff >= 1, "corpus must cover user-requested doff priority");
    assert!(blocked >= 7, "corpus needs diverse negative controls");
}
