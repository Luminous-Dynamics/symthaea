// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/full_duplex_turn_controller.rs"]
mod full_duplex_turn_controller;

use full_duplex_turn_controller::*;

#[test]
fn user_interruption_requests_yield_and_records_ack_latency_without_auto_resume() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.assistant_begin("turn-a", 100).unwrap();

    let actions = controller.user_speech_start("utt-a", 120).unwrap();
    assert!(actions.iter().any(|action| matches!(
        action,
        FullDuplexControlActionV1::RequestAssistantYield {
            turn_id,
            reason: FullDuplexYieldReasonV1::UserInterruption,
            requested_at_ns: 120,
        } if turn_id == "turn-a"
    )));
    assert!(matches!(
        controller.assistant_output(),
        FullDuplexAssistantOutputStateV1::YieldRequested {
            reason: FullDuplexYieldReasonV1::UserInterruption,
            ..
        }
    ));

    controller.acknowledge_output_stopped("turn-a", 135).unwrap();
    let receipt = controller.last_interruption_receipt().unwrap();
    assert_eq!(receipt.latency_ns, 15);
    assert!(matches!(
        controller.assistant_output(),
        FullDuplexAssistantOutputStateV1::Silent
    ));
    assert!(matches!(
        controller.user_floor(),
        FullDuplexUserFloorStateV1::Speaking { .. }
    ));

    assert_eq!(
        controller.assistant_begin("turn-b", 140),
        Err(FullDuplexTurnErrorV1::UserFloorOccupied)
    );
    controller.user_speech_end("utt-a", 150).unwrap();
    controller.assistant_begin("turn-b", 160).unwrap();
    assert!(matches!(
        controller.assistant_output(),
        FullDuplexAssistantOutputStateV1::Speaking { turn_id, .. } if turn_id == "turn-b"
    ));
}

#[test]
fn classified_backchannel_does_not_steal_floor_or_request_yield() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.assistant_begin("turn-a", 100).unwrap();
    let action = controller.user_backchannel("bc-a", 110).unwrap();
    assert_eq!(
        action,
        FullDuplexControlActionV1::BackchannelObserved {
            event_id: "bc-a".into(),
            assistant_output_active: true,
        }
    );
    assert!(matches!(
        controller.assistant_output(),
        FullDuplexAssistantOutputStateV1::Speaking { turn_id, .. } if turn_id == "turn-a"
    ));
    assert!(matches!(controller.user_floor(), FullDuplexUserFloorStateV1::Silent));
}

#[test]
fn explicit_stop_latches_and_measures_output_stop_latency() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.assistant_begin("turn-a", 100).unwrap();
    let actions = controller.user_stop("stop-a", 125).unwrap();
    assert!(actions.iter().any(|action| matches!(
        action,
        FullDuplexControlActionV1::RequestAssistantYield {
            reason: FullDuplexYieldReasonV1::ExplicitStop,
            requested_at_ns: 125,
            ..
        }
    )));
    controller.acknowledge_output_stopped("turn-a", 132).unwrap();
    let receipt = controller.last_stop_receipt().unwrap();
    assert_eq!(receipt.event_id, "stop-a");
    assert_eq!(receipt.latency_ns, 7);
    assert_eq!(receipt.turn_id.as_deref(), Some("turn-a"));
    assert_eq!(
        controller.assistant_begin("turn-b", 140),
        Err(FullDuplexTurnErrorV1::StopIsLatched)
    );

    let repeated = controller.user_stop("stop-a", 145).unwrap();
    assert_eq!(
        repeated,
        vec![FullDuplexControlActionV1::StopLatched {
            event_id: "stop-a".into(),
            newly_latched: false,
        }]
    );
}

#[test]
fn stop_while_already_silent_has_zero_output_latency() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.user_stop("stop-a", 100).unwrap();
    let receipt = controller.last_stop_receipt().unwrap();
    assert_eq!(receipt.latency_ns, 0);
    assert!(receipt.turn_id.is_none());
}

#[test]
fn slowdown_latches_until_explicit_compliant_plan_acknowledgement() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    let first = controller.user_slow_down("slow-a", 100).unwrap();
    assert_eq!(
        first,
        FullDuplexControlActionV1::DeescalationRequired {
            event_id: "slow-a".into(),
            newly_latched: true,
        }
    );
    let repeated = controller.user_slow_down("slow-a", 105).unwrap();
    assert_eq!(
        repeated,
        FullDuplexControlActionV1::DeescalationRequired {
            event_id: "slow-a".into(),
            newly_latched: false,
        }
    );

    let permit = controller.assistant_begin("turn-a", 110).unwrap();
    assert!(matches!(
        permit,
        FullDuplexControlActionV1::AssistantSpeechPermitted(
            FullDuplexAssistantSpeechPermitV1 {
                deescalation_required: true,
                ..
            }
        )
    ));
    controller.assistant_end("turn-a", 120).unwrap();

    let ack = controller
        .acknowledge_deescalation("plan:deescalated-1", 130)
        .unwrap();
    assert_eq!(
        ack,
        FullDuplexControlActionV1::DeescalationAcknowledged {
            plan_ref: "plan:deescalated-1".into(),
            latency_ns: 30,
        }
    );
    assert!(!controller.deescalation_required());
    assert_eq!(
        controller.last_deescalation_receipt().unwrap().latency_ns,
        30
    );
}

#[test]
fn assistant_cannot_begin_while_user_floor_is_active() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.user_speech_start("utt-a", 100).unwrap();
    assert_eq!(
        controller.assistant_begin("turn-a", 110),
        Err(FullDuplexTurnErrorV1::UserFloorOccupied)
    );
}

#[test]
fn rejected_event_does_not_advance_clock_or_mutate_valid_floor_state() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.user_speech_start("utt-a", 100).unwrap();
    assert_eq!(
        controller.user_speech_end("wrong-id", 200),
        Err(FullDuplexTurnErrorV1::UserUtteranceMismatch)
    );
    controller.user_speech_end("utt-a", 150).unwrap();
    assert!(matches!(controller.user_floor(), FullDuplexUserFloorStateV1::Silent));
}

#[test]
fn response_latency_is_measured_from_user_floor_release() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.user_speech_start("utt-a", 100).unwrap();
    controller.user_speech_end("utt-a", 140).unwrap();
    let permit = controller.assistant_begin("turn-a", 175).unwrap();
    assert!(matches!(
        permit,
        FullDuplexControlActionV1::AssistantSpeechPermitted(
            FullDuplexAssistantSpeechPermitV1 {
                response_latency_ns: Some(35),
                ..
            }
        )
    ));
}

#[test]
fn natural_end_is_not_a_yield_ack_and_pending_yield_requires_ack_path() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.assistant_begin("turn-a", 100).unwrap();
    controller.user_speech_start("utt-a", 110).unwrap();
    assert_eq!(
        controller.assistant_end("turn-a", 120),
        Err(FullDuplexTurnErrorV1::YieldAcknowledgementRequired)
    );
    controller.acknowledge_output_stopped("turn-a", 125).unwrap();
}

#[test]
fn fresh_session_reset_is_explicit_and_requires_quiescent_channels() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.assistant_begin("turn-a", 100).unwrap();
    assert_eq!(
        controller.reset_session(2, 110),
        Err(FullDuplexTurnErrorV1::SessionResetUnsafe)
    );
    controller.user_stop("stop-a", 120).unwrap();
    controller.acknowledge_output_stopped("turn-a", 125).unwrap();
    controller.reset_session(2, 130).unwrap();
    assert_eq!(controller.session_epoch(), 2);
    assert!(controller.stop_latch().is_none());
    controller.assistant_begin("turn-a", 140).unwrap();
}

#[test]
fn non_monotonic_valid_event_is_rejected() {
    let mut controller = FullDuplexTurnControllerV1::new(1).unwrap();
    controller.user_backchannel("bc-a", 100).unwrap();
    assert_eq!(
        controller.user_backchannel("bc-b", 99),
        Err(FullDuplexTurnErrorV1::NonMonotonicTime)
    );
}
