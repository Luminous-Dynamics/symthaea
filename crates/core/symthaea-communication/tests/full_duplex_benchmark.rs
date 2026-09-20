// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/full_duplex_turn_controller.rs"]
mod full_duplex_turn_controller;
#[path = "../src/full_duplex_benchmark.rs"]
mod full_duplex_benchmark;

use full_duplex_benchmark::*;

fn interruption_budget(max: u64) -> FullDuplexLatencyBudgetV1 {
    FullDuplexLatencyBudgetV1 {
        max_interruption_yield_latency_ns: Some(max),
        ..FullDuplexLatencyBudgetV1::default()
    }
}

fn stop_budget(max: u64) -> FullDuplexLatencyBudgetV1 {
    FullDuplexLatencyBudgetV1 {
        max_stop_latency_ns: Some(max),
        ..FullDuplexLatencyBudgetV1::default()
    }
}

fn deescalation_budget(max: u64) -> FullDuplexLatencyBudgetV1 {
    FullDuplexLatencyBudgetV1 {
        max_deescalation_ack_latency_ns: Some(max),
        ..FullDuplexLatencyBudgetV1::default()
    }
}

fn response_budget(max: u64) -> FullDuplexLatencyBudgetV1 {
    FullDuplexLatencyBudgetV1 {
        max_response_latency_ns: Some(max),
        ..FullDuplexLatencyBudgetV1::default()
    }
}

#[test]
fn interruption_yield_is_measured_independently_from_response_quality() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "interrupt-yield",
        1,
        FullDuplexBenchmarkScenarioV1::InterruptionYield,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 200,
            },
            FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped {
                turn_id: "turn-a".into(),
                at_ns: 250,
            },
            FullDuplexBenchmarkEventV1::UserSpeechEnd {
                utterance_id: "utt-a".into(),
                at_ns: 300,
            },
        ],
        interruption_budget(60),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.trace_status, FullDuplexTraceStatusV1::Complete);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(receipt.latency.interruption_yield_latency_ns, Some(50));
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::WithinBudget
    );
    assert!(receipt.final_assistant_output_silent);
}

#[test]
fn measured_latency_can_exceed_budget_without_becoming_semantic_failure() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "interrupt-over-budget",
        1,
        FullDuplexBenchmarkScenarioV1::InterruptionYield,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 200,
            },
            FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped {
                turn_id: "turn-a".into(),
                at_ns: 250,
            },
            FullDuplexBenchmarkEventV1::UserSpeechEnd {
                utterance_id: "utt-a".into(),
                at_ns: 300,
            },
        ],
        interruption_budget(20),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::Exceeded
    );
    assert_eq!(
        receipt.latency_violations,
        vec![FullDuplexLatencyViolationV1::InterruptionYield]
    );
}

#[test]
fn missing_requested_latency_is_not_a_budget_pass() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "interrupt-incomplete-with-budget",
        1,
        FullDuplexBenchmarkScenarioV1::InterruptionYield,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 200,
            },
        ],
        interruption_budget(60),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(
        receipt.semantic_status,
        FullDuplexSemanticStatusV1::NotEstablished
    );
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::NotEstablished
    );
    assert!(receipt.latency_violations.is_empty());
}

#[test]
fn explicit_stop_latches_and_measures_output_stop_latency() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "stop",
        1,
        FullDuplexBenchmarkScenarioV1::ExplicitStop,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserStop {
                event_id: "stop-a".into(),
                at_ns: 150,
            },
            FullDuplexBenchmarkEventV1::AcknowledgeOutputStopped {
                turn_id: "turn-a".into(),
                at_ns: 180,
            },
        ],
        stop_budget(40),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(receipt.latency.stop_latency_ns, Some(30));
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::WithinBudget
    );
    assert!(receipt.final_stop_latched);
    assert!(receipt.final_assistant_output_silent);
}

#[test]
fn backchannel_does_not_steal_the_floor() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "backchannel",
        1,
        FullDuplexBenchmarkScenarioV1::BackchannelContinuity,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserBackchannel {
                event_id: "bc-a".into(),
                at_ns: 120,
            },
        ],
        FullDuplexLatencyBudgetV1::default(),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(receipt.backchannel_count, 1);
    assert!(!receipt.final_assistant_output_silent);
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::NotEvaluated
    );
}

#[test]
fn slowdown_acknowledgement_is_measured_without_becoming_stop() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "slowdown",
        1,
        FullDuplexBenchmarkScenarioV1::DeescalationAcknowledgement,
        vec![
            FullDuplexBenchmarkEventV1::UserSlowDown {
                event_id: "slow-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::AcknowledgeDeescalation {
                plan_ref: "plan:ease-down".into(),
                at_ns: 160,
            },
        ],
        deescalation_budget(80),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(receipt.latency.deescalation_ack_latency_ns, Some(60));
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::WithinBudget
    );
    assert!(!receipt.final_deescalation_required);
    assert!(!receipt.final_stop_latched);
}

#[test]
fn response_latency_is_measured_from_floor_release_to_assistant_start() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "response-latency",
        1,
        FullDuplexBenchmarkScenarioV1::ResponseLatency,
        vec![
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserSpeechEnd {
                utterance_id: "utt-a".into(),
                at_ns: 150,
            },
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 220,
            },
        ],
        response_budget(90),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.semantic_status, FullDuplexSemanticStatusV1::Satisfied);
    assert_eq!(receipt.latency.response_latency_ns, Some(70));
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::WithinBudget
    );
    assert!(receipt.final_user_floor_silent);
}

#[test]
fn incomplete_trace_without_budget_is_not_a_semantic_pass() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "interrupt-incomplete",
        1,
        FullDuplexBenchmarkScenarioV1::InterruptionYield,
        vec![
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 200,
            },
        ],
        FullDuplexLatencyBudgetV1::default(),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(
        receipt.trace_status,
        FullDuplexTraceStatusV1::SemanticallyIncomplete
    );
    assert_eq!(
        receipt.semantic_status,
        FullDuplexSemanticStatusV1::NotEstablished
    );
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::NotEvaluated
    );
}

#[test]
fn controller_rejection_is_trace_failure_not_latency_failure() {
    let case = FullDuplexBenchmarkCaseV1::new(
        "invalid-overlap",
        1,
        FullDuplexBenchmarkScenarioV1::MixedInteraction,
        vec![
            FullDuplexBenchmarkEventV1::UserSpeechStart {
                utterance_id: "utt-a".into(),
                at_ns: 100,
            },
            FullDuplexBenchmarkEventV1::AssistantBegin {
                turn_id: "turn-a".into(),
                at_ns: 110,
            },
        ],
        response_budget(90),
    )
    .unwrap();
    let receipt = run_full_duplex_benchmark_case_v1(&case);
    assert_eq!(receipt.trace_status, FullDuplexTraceStatusV1::RuntimeRejected);
    assert_eq!(
        receipt.semantic_status,
        FullDuplexSemanticStatusV1::NotEstablished
    );
    assert_eq!(
        receipt.latency_budget_status,
        FullDuplexLatencyBudgetStatusV1::NotEvaluated
    );
    assert_eq!(receipt.accepted_events, 1);
    assert_eq!(receipt.rejected_step_index, Some(1));
    assert!(receipt.rejected_error.is_some());
}

#[test]
fn case_commitment_binds_script_and_latency_policy() {
    let events = vec![
        FullDuplexBenchmarkEventV1::UserSpeechStart {
            utterance_id: "utt-a".into(),
            at_ns: 100,
        },
        FullDuplexBenchmarkEventV1::UserSpeechEnd {
            utterance_id: "utt-a".into(),
            at_ns: 150,
        },
        FullDuplexBenchmarkEventV1::AssistantBegin {
            turn_id: "turn-a".into(),
            at_ns: 220,
        },
    ];
    let a = FullDuplexBenchmarkCaseV1::new(
        "same",
        1,
        FullDuplexBenchmarkScenarioV1::ResponseLatency,
        events.clone(),
        response_budget(90),
    )
    .unwrap();
    let b = FullDuplexBenchmarkCaseV1::new(
        "same",
        1,
        FullDuplexBenchmarkScenarioV1::ResponseLatency,
        events,
        response_budget(10),
    )
    .unwrap();
    assert_ne!(a.commitment_v1(), b.commitment_v1());
}

#[test]
fn non_monotonic_fixture_is_rejected_before_execution() {
    let result = FullDuplexBenchmarkCaseV1::new(
        "bad-time",
        1,
        FullDuplexBenchmarkScenarioV1::MixedInteraction,
        vec![
            FullDuplexBenchmarkEventV1::UserBackchannel {
                event_id: "a".into(),
                at_ns: 200,
            },
            FullDuplexBenchmarkEventV1::UserBackchannel {
                event_id: "b".into(),
                at_ns: 100,
            },
        ],
        FullDuplexLatencyBudgetV1::default(),
    );
    assert_eq!(
        result,
        Err(FullDuplexBenchmarkDefinitionErrorV1::NonMonotonicEventTime)
    );
}
