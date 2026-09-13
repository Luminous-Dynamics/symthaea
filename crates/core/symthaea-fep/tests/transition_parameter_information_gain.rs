//! Construct-validity regressions for transition-parameter information gain V1.
//!
//! These tests qualify a measurement only. They do not connect the estimator to
//! expected-free-energy scoring or policy selection.

use symthaea_fep::{
    ModelConfidenceTracker, TransitionParameterInformationGainError,
    dirichlet_transition_parameter_information_gain_nats_v1,
};

fn tracker(actions: usize, states: usize) -> ModelConfidenceTracker {
    ModelConfidenceTracker::new(actions, states, states, 0.99, 0.1)
}

#[test]
fn unseen_binary_transition_matches_closed_form() {
    let measured = dirichlet_transition_parameter_information_gain_nats_v1(&[0, 0])
        .expect("two-outcome row is valid");
    let expected = 2.0_f64.ln() - 0.5;

    assert!(
        (measured - expected).abs() < 1e-12,
        "Dirichlet(1,1) one-step information gain should be ln(2)-1/2: measured={measured}, expected={expected}"
    );
}

#[test]
fn outcome_label_permutation_cannot_change_information_gain() {
    let a = dirichlet_transition_parameter_information_gain_nats_v1(&[7, 1, 0, 3])
        .expect("row is valid");
    let b = dirichlet_transition_parameter_information_gain_nats_v1(&[3, 0, 7, 1])
        .expect("permuted row is valid");
    let c = dirichlet_transition_parameter_information_gain_nats_v1(&[1, 7, 3, 0])
        .expect("permuted row is valid");

    assert!((a - b).abs() < 1e-12);
    assert!((a - c).abs() < 1e-12);
}

#[test]
fn growing_symmetric_evidence_reduces_expected_information_gain() {
    let unseen = dirichlet_transition_parameter_information_gain_nats_v1(&[0, 0, 0, 0])
        .expect("row is valid");
    let one_each = dirichlet_transition_parameter_information_gain_nats_v1(&[1, 1, 1, 1])
        .expect("row is valid");
    let ten_each = dirichlet_transition_parameter_information_gain_nats_v1(&[10, 10, 10, 10])
        .expect("row is valid");
    let thousand_each =
        dirichlet_transition_parameter_information_gain_nats_v1(&[1000, 1000, 1000, 1000])
            .expect("row is valid");

    assert!(unseen > one_each);
    assert!(one_each > ten_each);
    assert!(ten_each > thousand_each);
    assert!(thousand_each > 0.0);
}

#[test]
fn confirmed_transition_evidence_is_action_and_source_state_local() {
    let mut evidence = tracker(3, 4);
    let target_before = evidence
        .transition_parameter_information_gain_nats_v1(1, 2)
        .expect("target row is valid");
    let other_action_before = evidence
        .transition_parameter_information_gain_nats_v1(0, 2)
        .expect("comparison action row is valid");
    let other_state_before = evidence
        .transition_parameter_information_gain_nats_v1(1, 1)
        .expect("comparison source-state row is valid");

    for _ in 0..20 {
        evidence.update_transition(1, 2, 3);
    }

    let target_after = evidence
        .transition_parameter_information_gain_nats_v1(1, 2)
        .expect("target row is valid");
    let other_action_after = evidence
        .transition_parameter_information_gain_nats_v1(0, 2)
        .expect("comparison action row is valid");
    let other_state_after = evidence
        .transition_parameter_information_gain_nats_v1(1, 1)
        .expect("comparison source-state row is valid");

    assert!(target_after < target_before);
    assert_eq!(other_action_after, other_action_before);
    assert_eq!(other_state_after, other_state_before);
}

#[test]
fn information_gain_query_is_side_effect_free() {
    let mut evidence = tracker(2, 3);
    evidence.update_transition(1, 0, 2);
    evidence.update_transition(1, 0, 2);

    let counts_before = evidence.transition_counts.clone();
    let confidence_before = evidence.transition_confidence.clone();
    let likelihood_counts_before = evidence.likelihood_counts.clone();
    let likelihood_confidence_before = evidence.likelihood_confidence.clone();

    let first = evidence
        .transition_parameter_information_gain_nats_v1(1, 0)
        .expect("row is valid");
    let second = evidence
        .transition_parameter_information_gain_nats_v1(1, 0)
        .expect("repeated query is valid");

    assert_eq!(first, second);
    assert_eq!(evidence.transition_counts, counts_before);
    assert_eq!(evidence.transition_confidence, confidence_before);
    assert_eq!(evidence.likelihood_counts, likelihood_counts_before);
    assert_eq!(evidence.likelihood_confidence, likelihood_confidence_before);
}

#[test]
fn malformed_evidence_identity_fails_closed_instead_of_clamping() {
    let mut evidence = tracker(2, 3);

    assert_eq!(
        evidence.transition_parameter_information_gain_nats_v1(2, 0),
        Err(TransitionParameterInformationGainError::ActionOutOfRange {
            action: 2,
            num_actions: 2,
        })
    );
    assert_eq!(
        evidence.transition_parameter_information_gain_nats_v1(1, 3),
        Err(TransitionParameterInformationGainError::FromStateOutOfRange {
            from_state: 3,
            state_count: 3,
        })
    );

    evidence.transition_counts[0][1].clear();
    assert_eq!(
        evidence.transition_parameter_information_gain_nats_v1(0, 1),
        Err(TransitionParameterInformationGainError::EmptyOutcomeDomain)
    );
}

#[test]
fn action_permutation_with_evidence_permutation_preserves_values() {
    let mut original = tracker(3, 4);
    original.transition_counts[0][1] = vec![8, 2, 0, 1];
    original.transition_counts[1][1] = vec![0, 5, 4, 2];
    original.transition_counts[2][1] = vec![3, 3, 3, 3];

    let mut permuted = tracker(3, 4);
    permuted.transition_counts[0][1] = original.transition_counts[2][1].clone();
    permuted.transition_counts[1][1] = original.transition_counts[0][1].clone();
    permuted.transition_counts[2][1] = original.transition_counts[1][1].clone();

    assert_eq!(
        original
            .transition_parameter_information_gain_nats_v1(0, 1)
            .unwrap(),
        permuted
            .transition_parameter_information_gain_nats_v1(1, 1)
            .unwrap()
    );
    assert_eq!(
        original
            .transition_parameter_information_gain_nats_v1(1, 1)
            .unwrap(),
        permuted
            .transition_parameter_information_gain_nats_v1(2, 1)
            .unwrap()
    );
    assert_eq!(
        original
            .transition_parameter_information_gain_nats_v1(2, 1)
            .unwrap(),
        permuted
            .transition_parameter_information_gain_nats_v1(0, 1)
            .unwrap()
    );
}

#[test]
fn serialization_round_trip_preserves_information_gain_exactly() {
    let mut evidence = tracker(2, 4);
    for _ in 0..7 {
        evidence.update_transition(0, 1, 2);
    }
    for _ in 0..3 {
        evidence.update_transition(0, 1, 3);
    }
    for _ in 0..11 {
        evidence.update_transition(1, 2, 0);
    }

    let before_01 = evidence
        .transition_parameter_information_gain_nats_v1(0, 1)
        .unwrap();
    let before_12 = evidence
        .transition_parameter_information_gain_nats_v1(1, 2)
        .unwrap();

    let encoded = serde_json::to_string(&evidence).expect("tracker should serialize");
    let restored: ModelConfidenceTracker =
        serde_json::from_str(&encoded).expect("tracker should deserialize");

    assert_eq!(
        restored
            .transition_parameter_information_gain_nats_v1(0, 1)
            .unwrap(),
        before_01
    );
    assert_eq!(
        restored
            .transition_parameter_information_gain_nats_v1(1, 2)
            .unwrap(),
        before_12
    );
}

#[test]
fn extreme_counts_remain_finite_and_non_negative() {
    let value = dirichlet_transition_parameter_information_gain_nats_v1(&[
        u64::MAX,
        u64::MAX - 1,
        u64::MAX / 2,
        0,
    ])
    .expect("u64 evidence range should remain representable in V1");

    assert!(value.is_finite());
    assert!(value >= 0.0);
}
