#![cfg(feature = "symthaea-backend")]

//! Component-level AE-1 theorem for Symthaea's production active-inference engine.
//!
//! Butlin AE-1 requires more than action diversity: the system should learn from
//! feedback, pursue goals, and flexibly respond to competing goals. These tests
//! exercise those properties directly using the production `ActiveInferenceEngine`.
//! Full cognitive-loop integration and independent replication remain separate.

use symthaea::consciousness::predictive_processing::ActiveInferenceEngine;
use symthaea::hdc::binary_hv::BinaryHV;

fn goal_a() -> BinaryHV {
    BinaryHV::random(0xAE10_0001)
}

fn goal_b() -> BinaryHV {
    BinaryHV::random(0xAE10_0002)
}

fn motor_a() -> BinaryHV {
    BinaryHV::random(0xAE10_1001)
}

fn motor_b() -> BinaryHV {
    BinaryHV::random(0xAE10_1002)
}

fn selected_action(engine: &mut ActiveInferenceEngine, state: &BinaryHV) -> u32 {
    engine
        .select_action(state)
        .expect("test engine has goals and actions")
        .id
}

fn two_action_engine(goal: BinaryHV) -> ActiveInferenceEngine {
    let mut engine = ActiveInferenceEngine::new();
    engine.add_goal(goal, 1.0, 1);
    engine.add_action(motor_a(), goal_a());
    engine.add_action(motor_b(), goal_b());
    engine
}

#[test]
fn action_selection_pursues_the_active_goal() {
    let neutral_state = BinaryHV::random(0xAE10_F000);

    let mut pursue_a = two_action_engine(goal_a());
    let mut pursue_b = two_action_engine(goal_b());

    assert_eq!(
        selected_action(&mut pursue_a, &neutral_state),
        0,
        "engine did not select the action whose expected outcome matches goal A",
    );
    assert_eq!(
        selected_action(&mut pursue_b, &neutral_state),
        1,
        "engine did not select the action whose expected outcome matches goal B",
    );
}

fn competing_goal_engine(a_precision: f64, b_precision: f64) -> ActiveInferenceEngine {
    let mut engine = ActiveInferenceEngine::new();
    engine.add_goal(goal_a(), a_precision, 1);
    engine.add_goal(goal_b(), b_precision, 1);
    engine.add_action(motor_a(), goal_a());
    engine.add_action(motor_b(), goal_b());
    engine
}

#[test]
fn reversing_competing_goal_precision_reverses_action_choice() {
    let neutral_state = BinaryHV::random(0xAE10_F001);

    let mut favor_a = competing_goal_engine(1.0, 0.05);
    let mut favor_b = competing_goal_engine(0.05, 1.0);

    assert_eq!(
        selected_action(&mut favor_a, &neutral_state),
        0,
        "high-precision goal A did not dominate policy selection",
    );
    assert_eq!(
        selected_action(&mut favor_b, &neutral_state),
        1,
        "high-precision goal B did not dominate policy selection",
    );
}

#[test]
fn negative_feedback_changes_policy_and_positive_feedback_rescues_it() {
    let target = BinaryHV::random(0xAE10_2001);
    let neutral_state = BinaryHV::random(0xAE10_F002);
    let mut engine = ActiveInferenceEngine::new();

    engine.add_goal(target, 1.0, 1);
    // Identical expected outcomes make the initial pragmatic and action-cost
    // terms equal. With equal prior success rates the stable tie resolves to
    // action 0; subsequent policy movement is therefore attributable to
    // feedback-updated success history rather than changed goals/outcomes.
    engine.add_action(motor_a(), target);
    engine.add_action(motor_b(), target);

    assert_eq!(selected_action(&mut engine, &neutral_state), 0);

    // Repeated failure increases action 0's epistemic/uncertainty cost.
    for _ in 0..12 {
        engine.learn_outcome(0, &target, false);
    }
    assert_eq!(
        selected_action(&mut engine, &neutral_state),
        1,
        "negative feedback did not shift policy away from the failing action",
    );

    // Rescue the original policy using feedback only; goal, current state and
    // expected outcome remain unchanged.
    for _ in 0..16 {
        engine.learn_outcome(0, &target, true);
    }
    assert_eq!(
        selected_action(&mut engine, &neutral_state),
        0,
        "positive feedback did not restore preference for the rehabilitated action",
    );
}

#[test]
fn feedback_for_one_action_does_not_require_mutating_the_other_action() {
    let target = BinaryHV::random(0xAE10_2002);
    let neutral_state = BinaryHV::random(0xAE10_F003);
    let mut engine = ActiveInferenceEngine::new();

    engine.add_goal(target, 1.0, 1);
    engine.add_action(motor_a(), target);
    engine.add_action(motor_b(), target);

    for _ in 0..12 {
        engine.learn_outcome(0, &target, false);
    }

    // Action 1 receives no fabricated success feedback. It becomes preferable
    // only because action 0's learned reliability falls.
    assert_eq!(selected_action(&mut engine, &neutral_state), 1);
}

#[test]
fn component_scope_refuses_ae1_functional_promotion() {
    const CLAIM_SCOPE: &str =
        "component minimal-agency theorem only; no full CognitiveLoopService consequence; no AE-1 tier promotion; no consciousness claim";
    assert!(CLAIM_SCOPE.contains("no AE-1 tier promotion"));
    assert!(CLAIM_SCOPE.contains("no consciousness claim"));
}
