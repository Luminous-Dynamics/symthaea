// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public-interface scenarios for the report-only assistive-mobility effort policy.
//!
//! These scenarios validate policy semantics in software only. They do not
//! establish physical safety, clinical suitability, physiological effort, or
//! authorization to control hardware.

use symthaea_core::effort_policy::{
    evaluate_effort_request, EffortDecisionStatusV1, EffortEvidenceStatusV1, EffortFieldV1,
    EffortFindingV1, EffortModeV1, EffortRequestV1, EFFORT_POLICY_SCHEMA_VERSION,
};

struct Scenario {
    id: &'static str,
    request: EffortRequestV1,
    expected_status: EffortDecisionStatusV1,
    expected_finding: Option<EffortFindingV1>,
}

fn request_for(mode: EffortModeV1) -> EffortRequestV1 {
    let (explicit_training_opt_in, profile_training_eligible, requested_resistance) =
        if mode == EffortModeV1::TrainingResistive {
            (true, true, 0.3)
        } else {
            (false, false, 0.0)
        };

    EffortRequestV1 {
        schema_version: EFFORT_POLICY_SCHEMA_VERSION,
        session_id: "session-open-mobility-001".to_owned(),
        active_session_id: "session-open-mobility-001".to_owned(),
        selected_profile_id: "profile-example-001".to_owned(),
        active_profile_id: "profile-example-001".to_owned(),
        mode,
        explicit_training_opt_in,
        profile_training_eligible,
        effort_evidence: EffortEvidenceStatusV1::Fresh,
        user_requested_stop: false,
        user_requested_recovery: false,
        user_requested_more_assistance: false,
        resistance_increase_requested: false,
        requested_assistance: 0.4,
        assistance_limit: 0.8,
        requested_resistance,
        resistance_limit: 0.5,
    }
}

fn corpus() -> Vec<Scenario> {
    let mut cases = Vec::new();

    cases.push(Scenario {
        id: "MOB-001-valid-mobility",
        request: request_for(EffortModeV1::MobilityConserve),
        expected_status: EffortDecisionStatusV1::AdmissibleForSimulation,
        expected_finding: None,
    });

    cases.push(Scenario {
        id: "MOB-002-valid-assist-as-needed",
        request: request_for(EffortModeV1::AssistAsNeeded),
        expected_status: EffortDecisionStatusV1::AdmissibleForSimulation,
        expected_finding: None,
    });

    cases.push(Scenario {
        id: "TRN-001-explicit-opt-in",
        request: request_for(EffortModeV1::TrainingResistive),
        expected_status: EffortDecisionStatusV1::AdmissibleForSimulation,
        expected_finding: None,
    });

    let mut unauthorized_resistance = request_for(EffortModeV1::MobilityConserve);
    unauthorized_resistance.requested_resistance = 0.2;
    cases.push(Scenario {
        id: "NEG-001-resistance-outside-training",
        request: unauthorized_resistance,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::ResistanceRequiresTrainingMode),
    });

    let mut no_opt_in = request_for(EffortModeV1::TrainingResistive);
    no_opt_in.explicit_training_opt_in = false;
    cases.push(Scenario {
        id: "NEG-002-training-without-opt-in",
        request: no_opt_in,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::TrainingOptInRequired),
    });

    let mut profile_mismatch = request_for(EffortModeV1::TrainingResistive);
    profile_mismatch.active_profile_id = "previous-profile".to_owned();
    cases.push(Scenario {
        id: "NEG-003-profile-mismatch",
        request: profile_mismatch,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::ProfileMismatch),
    });

    let mut session_mismatch = request_for(EffortModeV1::TrainingResistive);
    session_mismatch.active_session_id = "previous-session".to_owned();
    cases.push(Scenario {
        id: "NEG-004-stale-session",
        request: session_mismatch,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::SessionMismatch),
    });

    let mut stale_progression = request_for(EffortModeV1::TrainingResistive);
    stale_progression.effort_evidence = EffortEvidenceStatusV1::Stale;
    stale_progression.resistance_increase_requested = true;
    cases.push(Scenario {
        id: "NEG-005-stale-evidence-cannot-increase-resistance",
        request: stale_progression,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::ResistanceIncreaseRequiresFreshEvidence),
    });

    let mut no_voluntary_signal = request_for(EffortModeV1::TrainingResistive);
    no_voluntary_signal.effort_evidence = EffortEvidenceStatusV1::NoVoluntarySignal;
    cases.push(Scenario {
        id: "MOB-003-no-voluntary-signal-is-not-punitive",
        request: no_voluntary_signal,
        expected_status: EffortDecisionStatusV1::AdmissibleForSimulation,
        expected_finding: None,
    });

    let mut stop_preempts_training = request_for(EffortModeV1::TrainingResistive);
    stop_preempts_training.user_requested_stop = true;
    cases.push(Scenario {
        id: "PRI-001-stop-preempts-resistance",
        request: stop_preempts_training,
        expected_status: EffortDecisionStatusV1::StopRequired,
        expected_finding: Some(EffortFindingV1::UserRequestedStop),
    });

    let recovery = request_for(EffortModeV1::RecoveryHighSupport);
    cases.push(Scenario {
        id: "PRI-002-recovery-mode",
        request: recovery,
        expected_status: EffortDecisionStatusV1::RecoveryRequired,
        expected_finding: Some(EffortFindingV1::UserRequestedRecovery),
    });

    let mut non_finite = request_for(EffortModeV1::TrainingResistive);
    non_finite.requested_assistance = f32::NAN;
    cases.push(Scenario {
        id: "NEG-006-non-finite-assistance",
        request: non_finite,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::NonFiniteValue {
            field: EffortFieldV1::RequestedAssistance,
        }),
    });

    let mut above_limit = request_for(EffortModeV1::TrainingResistive);
    above_limit.requested_resistance = 0.7;
    cases.push(Scenario {
        id: "NEG-007-resistance-above-declared-limit",
        request: above_limit,
        expected_status: EffortDecisionStatusV1::Rejected,
        expected_finding: Some(EffortFindingV1::RequestedAboveLimit {
            field: EffortFieldV1::RequestedResistance,
        }),
    });

    cases
}

#[test]
fn public_effort_policy_scenario_corpus_matches_expected_dispositions() {
    let cases = corpus();
    assert!(cases.len() >= 10, "scenario corpus unexpectedly shrank");

    for scenario in &cases {
        let actual = evaluate_effort_request(&scenario.request);
        assert_eq!(
            actual.status, scenario.expected_status,
            "unexpected disposition for scenario {}: {:?}",
            scenario.id, actual.findings
        );
        if let Some(ref finding) = scenario.expected_finding {
            assert!(
                actual.findings.contains(finding),
                "scenario {} did not emit expected finding {:?}; actual findings: {:?}",
                scenario.id,
                finding,
                actual.findings
            );
        }
    }
}

#[test]
fn public_effort_policy_corpus_replays_deterministically() {
    let cases = corpus();
    for scenario in &cases {
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
        .iter()
        .filter(|case| {
            case.expected_status == EffortDecisionStatusV1::AdmissibleForSimulation
        })
        .count();
    let negative = cases
        .iter()
        .filter(|case| case.expected_status == EffortDecisionStatusV1::Rejected)
        .count();

    assert!(positive >= 3, "need valid mobility, participation, and training controls");
    assert!(negative >= 5, "need diverse invalid-input controls");
}
