// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-product and metamorphic properties for the report-only effort policy.
//!
//! These tests cover software policy semantics only. They do not establish
//! actuator safety, sensor authenticity, clinical suitability, or physiological benefit.

use serde::Deserialize;
use symthaea_core::effort_policy::{
    evaluate_effort_request, EffortDecisionStatusV1, EffortEvidenceStatusV1, EffortModeV1,
    EffortRequestV1,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioCorpus {
    schema_version: u16,
    scenarios: Vec<ScenarioSeed>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioSeed {
    id: String,
    request: EffortRequestV1,
}

fn valid_mobility_seed() -> EffortRequestV1 {
    let corpus: ScenarioCorpus = serde_json::from_str(include_str!(
        "fixtures/effort-policy-scenarios-v1.json"
    ))
    .expect("versioned effort-policy corpus parses");
    assert_eq!(corpus.schema_version, 1);
    corpus
        .scenarios
        .into_iter()
        .find(|scenario| scenario.id == "MOB-001-valid-mobility")
        .expect("corpus contains the valid mobility seed")
        .request
}

#[test]
fn exhaustive_control_cross_product_preserves_effort_policy_invariants() {
    let modes = [
        EffortModeV1::MobilityConserve,
        EffortModeV1::AssistAsNeeded,
        EffortModeV1::TrainingResistive,
        EffortModeV1::RecoveryHighSupport,
    ];
    let evidence_states = [
        EffortEvidenceStatusV1::Fresh,
        EffortEvidenceStatusV1::Unavailable,
        EffortEvidenceStatusV1::NoVoluntarySignal,
        EffortEvidenceStatusV1::Stale,
        EffortEvidenceStatusV1::Conflicted,
        EffortEvidenceStatusV1::Invalid,
    ];
    let bools = [false, true];
    let resistance_values = [0.0_f32, 0.2_f32];
    let mut evaluated = 0_usize;

    for mode in modes {
        for evidence in evidence_states {
            for opt_in in bools {
                for profile_eligible in bools {
                    for stop in bools {
                        for recovery in bools {
                            for more_assistance in bools {
                                for increase in bools {
                                    for resistance in resistance_values {
                                        let mut request = valid_mobility_seed();
                                        request.mode = mode;
                                        request.effort_evidence = evidence;
                                        request.explicit_training_opt_in = opt_in;
                                        request.profile_training_eligible = profile_eligible;
                                        request.user_requested_stop = stop;
                                        request.user_requested_recovery = recovery;
                                        request.user_requested_more_assistance = more_assistance;
                                        request.resistance_increase_requested = increase;
                                        request.requested_resistance = resistance;
                                        request.resistance_limit = 0.5;

                                        let expected = if stop {
                                            EffortDecisionStatusV1::StopRequired
                                        } else if recovery || mode == EffortModeV1::RecoveryHighSupport {
                                            EffortDecisionStatusV1::RecoveryRequired
                                        } else if more_assistance {
                                            EffortDecisionStatusV1::SupportRequired
                                        } else {
                                            let prohibited_resistance =
                                                mode != EffortModeV1::TrainingResistive
                                                    && (resistance != 0.0 || increase);
                                            let invalid_training =
                                                mode == EffortModeV1::TrainingResistive
                                                    && (!opt_in
                                                        || !profile_eligible
                                                        || (increase
                                                            && evidence
                                                                != EffortEvidenceStatusV1::Fresh));
                                            if prohibited_resistance || invalid_training {
                                                EffortDecisionStatusV1::Rejected
                                            } else {
                                                EffortDecisionStatusV1::AdmissibleForSimulation
                                            }
                                        };

                                        let actual = evaluate_effort_request(&request);
                                        assert_eq!(
                                            actual.status, expected,
                                            "policy invariant mismatch at mode={mode:?}, evidence={evidence:?}, opt_in={opt_in}, profile_eligible={profile_eligible}, stop={stop}, recovery={recovery}, more_assistance={more_assistance}, increase={increase}, resistance={resistance}: {:?}",
                                            actual.findings
                                        );

                                        // Priority dispositions are never interpreted as
                                        // simulation admission, even when several user controls
                                        // are active in the same request.
                                        if stop || recovery || more_assistance
                                            || mode == EffortModeV1::RecoveryHighSupport
                                        {
                                            assert_ne!(
                                                actual.status,
                                                EffortDecisionStatusV1::AdmissibleForSimulation,
                                                "user stop/support/recovery intent cannot be admitted as an ordinary request"
                                            );
                                        }

                                        evaluated += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    assert_eq!(evaluated, 3_072, "cross-product coverage changed unexpectedly");
}

#[test]
fn user_control_priority_does_not_erase_policy_failure_evidence() {
    let mut request = valid_mobility_seed();
    request.mode = EffortModeV1::TrainingResistive;
    request.user_requested_stop = true;
    request.user_requested_recovery = true;
    request.user_requested_more_assistance = true;
    request.resistance_increase_requested = true;
    request.requested_resistance = 0.2;
    request.explicit_training_opt_in = false;
    request.profile_training_eligible = false;
    request.effort_evidence = EffortEvidenceStatusV1::Stale;
    request.active_session_id = "prior-session".to_owned();
    request.schema_version = 99;
    request.requested_assistance = f32::NAN;

    let stop = evaluate_effort_request(&request);
    assert_eq!(stop.status, EffortDecisionStatusV1::StopRequired);
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::UserRequestedStop
    )));
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::SessionMismatch
    )));
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::UnsupportedSchemaVersion { received: 99 }
    )));
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::NonFiniteValue { .. }
    )));
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::TrainingOptInRequired
    )));
    assert!(stop.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::ResistanceIncreaseRequiresFreshEvidence
    )));

    request.user_requested_stop = false;
    let recovery = evaluate_effort_request(&request);
    assert_eq!(recovery.status, EffortDecisionStatusV1::RecoveryRequired);

    request.user_requested_recovery = false;
    request.mode = EffortModeV1::TrainingResistive;
    let support = evaluate_effort_request(&request);
    assert_eq!(support.status, EffortDecisionStatusV1::SupportRequired);
    assert!(support.findings.iter().any(|finding| matches!(
        finding,
        symthaea_core::effort_policy::EffortFindingV1::SessionMismatch
    )));
}
