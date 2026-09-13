//! Construct-validity regressions for linear-Gaussian hidden-state salience V1.
//!
//! This suite qualifies a measurement only. It intentionally does not alter or
//! validate the live expected-free-energy policy path.

use symthaea_fep::generative_model::ActionTransitionPriorV1;
use symthaea_fep::{
    GenerativeModel, HiddenState, HiddenStateInformationGainError,
    linear_gaussian_hidden_state_information_gain_nats_v1,
};

fn two_state_model(actions: usize) -> GenerativeModel {
    let mut model = GenerativeModel::new(2, 2, actions);
    model.likelihood_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    model.observation_precision = 2.0;
    model.transition_precision = 1_000.0;
    model
}

fn asymmetric_state() -> HiddenState {
    let mut state = HiddenState::new(2);
    state.precision = vec![1.0, 4.0];
    state
}

fn identity_prior() -> ActionTransitionPriorV1 {
    ActionTransitionPriorV1::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
    )
}

fn mixing_prior() -> ActionTransitionPriorV1 {
    ActionTransitionPriorV1::new(
        vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        vec![0.0, 0.0],
    )
}

#[test]
fn one_dimensional_case_matches_closed_form() {
    let mut model = GenerativeModel::new(1, 1, 1);
    model.transition_matrices[0] = vec![vec![1.0]];
    model.likelihood_matrix = vec![vec![1.0]];
    model.transition_precision = 4.0;
    model.observation_precision = 3.0;

    let mut state = HiddenState::new(1);
    state.precision = vec![2.0];

    // Current variance = 1/2, process variance = 1/4, so predicted variance = 3/4.
    // I(s;o) = 1/2 ln(1 + observation_precision * predicted_variance).
    let expected = 0.5 * (1.0_f64 + 3.0 * 0.75).ln();
    let measured = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("valid one-dimensional model");

    assert!(
        (measured - expected).abs() < 1e-12,
        "measured={measured}, expected={expected}"
    );
}

#[test]
fn action_neutral_transition_prior_has_action_neutral_salience() {
    let model = two_state_model(4);
    let state = asymmetric_state();
    let first = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("action 0 should be valid");

    for action in 1..4 {
        let value = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, action)
            .expect("action should be valid");
        assert_eq!(value, first);
    }
}

#[test]
fn explicit_action_dynamics_can_create_distinct_salience() {
    let mut model = two_state_model(2);
    model
        .apply_action_transition_prior_v1(0, identity_prior())
        .expect("identity prior should apply");
    model
        .apply_action_transition_prior_v1(1, mixing_prior())
        .expect("mixing prior should apply");
    let state = asymmetric_state();

    let identity = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("identity action should be valid");
    let mixing = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 1)
        .expect("mixing action should be valid");

    assert_ne!(identity, mixing);
}

#[test]
fn action_permutation_with_dynamics_permutation_preserves_salience() {
    let mut original = two_state_model(2);
    original
        .apply_action_transition_prior_v1(0, identity_prior())
        .unwrap();
    original
        .apply_action_transition_prior_v1(1, mixing_prior())
        .unwrap();

    let mut permuted = original.clone();
    permuted.transition_matrices.swap(0, 1);
    permuted.transition_bias.swap(0, 1);
    let state = asymmetric_state();

    assert_eq!(
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &original, 0).unwrap(),
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &permuted, 1).unwrap()
    );
    assert_eq!(
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &original, 1).unwrap(),
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &permuted, 0).unwrap()
    );
}

#[test]
fn transition_bias_cannot_manufacture_salience() {
    let mut model = two_state_model(1);
    model
        .apply_action_transition_prior_v1(0, identity_prior())
        .unwrap();
    let state = asymmetric_state();
    let before = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("baseline should be valid");

    model.transition_bias[0] = vec![100.0, -50.0];
    let after = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("biased mean should still be valid");

    assert_eq!(after, before);
}

#[test]
fn more_precise_observations_have_more_hidden_state_information() {
    let mut low_precision = two_state_model(1);
    low_precision.observation_precision = 0.5;
    let mut high_precision = low_precision.clone();
    high_precision.observation_precision = 20.0;
    let state = asymmetric_state();

    let low = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &low_precision, 0)
        .unwrap();
    let high = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &high_precision, 0)
        .unwrap();

    assert!(high > low);
}

#[test]
fn more_precise_prior_state_has_less_information_left_to_gain() {
    let model = two_state_model(1);
    let mut uncertain = HiddenState::new(2);
    uncertain.precision = vec![0.5, 0.5];
    let mut certain = HiddenState::new(2);
    certain.precision = vec![100.0, 100.0];

    let uncertain_value =
        linear_gaussian_hidden_state_information_gain_nats_v1(&uncertain, &model, 0).unwrap();
    let certain_value =
        linear_gaussian_hidden_state_information_gain_nats_v1(&certain, &model, 0).unwrap();

    assert!(uncertain_value > certain_value);
}

#[test]
fn process_uncertainty_is_visible_to_future_observation_salience() {
    let mut noisy_transition = two_state_model(1);
    noisy_transition.transition_precision = 1.0;
    let mut precise_transition = noisy_transition.clone();
    precise_transition.transition_precision = 10_000.0;
    let state = asymmetric_state();

    let noisy =
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &noisy_transition, 0)
            .unwrap();
    let precise =
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &precise_transition, 0)
            .unwrap();

    assert!(noisy > precise);
}

#[test]
fn no_observation_channel_has_zero_salience() {
    let model = GenerativeModel::new(2, 0, 1);
    let state = HiddenState::new(2);
    assert_eq!(
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0).unwrap(),
        0.0
    );
}

#[test]
fn invalid_identity_shapes_and_precisions_fail_closed() {
    let state = asymmetric_state();
    let model = two_state_model(1);

    assert_eq!(
        linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 1),
        Err(HiddenStateInformationGainError::ActionOutOfRange {
            action: 1,
            num_actions: 1,
        })
    );

    let mut wrong_state = state.clone();
    wrong_state.precision.pop();
    assert_eq!(
        linear_gaussian_hidden_state_information_gain_nats_v1(&wrong_state, &model, 0),
        Err(HiddenStateInformationGainError::StatePrecisionLengthMismatch {
            expected: 2,
            actual: 1,
        })
    );

    let mut zero_precision_state = state.clone();
    zero_precision_state.precision[0] = 0.0;
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &zero_precision_state,
            &model,
            0
        ),
        Err(HiddenStateInformationGainError::InvalidStatePrecision { index: 0, .. })
    ));

    let mut invalid_transition_precision = model.clone();
    invalid_transition_precision.transition_precision = 0.0;
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &state,
            &invalid_transition_precision,
            0
        ),
        Err(HiddenStateInformationGainError::InvalidTransitionPrecision { .. })
    ));

    let mut invalid_observation_precision = model.clone();
    invalid_observation_precision.observation_precision = f64::NAN;
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &state,
            &invalid_observation_precision,
            0
        ),
        Err(HiddenStateInformationGainError::InvalidObservationPrecision { .. })
    ));

    let mut wrong_transition_shape = model.clone();
    wrong_transition_shape.transition_matrices[0][0].pop();
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &state,
            &wrong_transition_shape,
            0
        ),
        Err(HiddenStateInformationGainError::TransitionColumnCountMismatch { .. })
    ));

    let mut wrong_likelihood_shape = model.clone();
    wrong_likelihood_shape.likelihood_matrix[0].pop();
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &state,
            &wrong_likelihood_shape,
            0
        ),
        Err(HiddenStateInformationGainError::LikelihoodColumnCountMismatch { .. })
    ));

    let mut non_finite_transition = model.clone();
    non_finite_transition.transition_matrices[0][0][0] = f64::INFINITY;
    assert!(matches!(
        linear_gaussian_hidden_state_information_gain_nats_v1(
            &state,
            &non_finite_transition,
            0
        ),
        Err(HiddenStateInformationGainError::NonFiniteTransitionValue { .. })
    ));
}

#[test]
fn salience_query_is_side_effect_free() {
    let model = two_state_model(2);
    let state = asymmetric_state();
    let transitions_before = model.transition_matrices.clone();
    let biases_before = model.transition_bias.clone();
    let likelihood_before = model.likelihood_matrix.clone();
    let precision_before = state.precision.clone();

    let first = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("query should succeed");
    let second = linear_gaussian_hidden_state_information_gain_nats_v1(&state, &model, 0)
        .expect("repeated query should succeed");

    assert_eq!(first, second);
    assert_eq!(model.transition_matrices, transitions_before);
    assert_eq!(model.transition_bias, biases_before);
    assert_eq!(model.likelihood_matrix, likelihood_before);
    assert_eq!(state.precision, precision_before);
}
