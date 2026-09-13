//! Regression theorems for generic FEP action-model prior semantics.
//!
//! The generic core has no domain knowledge about integer action identifiers.
//! Therefore construction must be permutation-neutral: initial equality means
//! epistemic ignorance, not an even/odd or ordinal action ontology. Confirmed
//! transition learning may then differentiate individual action models.

use symthaea_fep::generative_model::{
    ACTION_TRANSITION_PRIOR_SCHEMA_V1, ActionTransitionPriorError, ActionTransitionPriorV1,
    GenerativeModel,
};
use symthaea_fep::types::{HiddenState, Observation};

const STATE_DIM: usize = 4;
const OBS_DIM: usize = 4;

fn asymmetric_state() -> HiddenState {
    let mut state = HiddenState::new(STATE_DIM);
    state.mean = vec![0.85, 0.10, 0.40, 0.25];
    state
}

fn one_hot_state(index: usize) -> HiddenState {
    let mut state = HiddenState::new(STATE_DIM);
    state.mean.fill(0.0);
    state.mean[index] = 1.0;
    state
}

fn observation() -> Observation {
    Observation::from_consciousness_state(0.7, 0.4, 0.6, 0.3)
}

fn explicit_prior() -> ActionTransitionPriorV1 {
    ActionTransitionPriorV1::new(
        vec![
            vec![0.80, 0.20, 0.00, 0.00],
            vec![0.10, 0.80, 0.10, 0.00],
            vec![0.00, 0.10, 0.80, 0.10],
            vec![0.00, 0.00, 0.20, 0.80],
        ],
        vec![0.02, -0.01, 0.00, 0.01],
    )
}

fn assert_rejected_without_mutation(
    model: &mut GenerativeModel,
    action: usize,
    prior: ActionTransitionPriorV1,
) -> ActionTransitionPriorError {
    let matrices_before = model.transition_matrices.clone();
    let biases_before = model.transition_bias.clone();
    let error = model
        .apply_action_transition_prior_v1(action, prior)
        .expect_err("invalid prior must fail closed");
    assert_eq!(model.transition_matrices, matrices_before);
    assert_eq!(model.transition_bias, biases_before);
    error
}

#[test]
fn generic_action_prior_is_permutation_neutral() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, 6);
    let first = &model.transition_matrices[0];

    for (action, transition) in model.transition_matrices.iter().enumerate().skip(1) {
        assert_eq!(
            transition, first,
            "action {action} received a different generic transition prior"
        );
    }

    let state = asymmetric_state();
    let first_prediction = model.predict_next_state(&state, 0).mean;
    for action in 1..model.num_actions {
        assert_eq!(
            model.predict_next_state(&state, action).mean,
            first_prediction,
            "renumbering an otherwise unknown action must not change its prior prediction"
        );
    }
}

#[test]
fn neutral_action_prior_rows_are_stochastic() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, 5);

    for (action, transition) in model.transition_matrices.iter().enumerate() {
        for (row_index, row) in transition.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "action {action} row {row_index} is not stochastic: sum={sum}"
            );
            assert!(
                row.iter().all(|p| *p >= 0.0 && *p <= 1.0),
                "action {action} row {row_index} contains an invalid probability"
            );
        }
    }
}

#[test]
fn confirmed_transition_learning_is_action_local() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, 3);
    let action_zero_before = model.transition_matrices[0].clone();
    let action_one_before = model.transition_matrices[1].clone();
    let action_two_before = model.transition_matrices[2].clone();

    let old_state = one_hot_state(0);
    let new_state = one_hot_state(1);
    model.learn_transition(&old_state, 1, &new_state, &observation());

    assert_eq!(model.transition_matrices[0], action_zero_before);
    assert_ne!(model.transition_matrices[1], action_one_before);
    assert_eq!(model.transition_matrices[2], action_two_before);
}

#[test]
fn distinct_confirmed_transitions_differentiate_action_models() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, 2);
    assert_eq!(model.transition_matrices[0], model.transition_matrices[1]);

    let old_state = one_hot_state(0);
    let action_zero_outcome = one_hot_state(1);
    let action_one_outcome = one_hot_state(2);

    model.learn_transition(&old_state, 0, &action_zero_outcome, &observation());
    model.learn_transition(&old_state, 1, &action_one_outcome, &observation());

    assert_ne!(model.transition_matrices[0], model.transition_matrices[1]);

    let predicted_zero = model.predict_next_state(&old_state, 0).mean;
    let predicted_one = model.predict_next_state(&old_state, 1).mean;
    assert_ne!(
        predicted_zero, predicted_one,
        "different confirmed consequences must become distinguishable predictions"
    );
}

#[test]
fn explicit_v1_prior_is_atomic_and_action_local() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, 3);
    let action_zero_before = model.transition_matrices[0].clone();
    let action_two_before = model.transition_matrices[2].clone();
    let prior = explicit_prior();

    model
        .apply_action_transition_prior_v1(1, prior.clone())
        .expect("valid V1 prior should apply");

    assert_eq!(model.transition_matrices[0], action_zero_before);
    assert_eq!(model.transition_matrices[1], prior.transition_matrix);
    assert_eq!(model.transition_bias[1], prior.transition_bias);
    assert_eq!(model.transition_matrices[2], action_two_before);
}

#[test]
fn invalid_v1_priors_fail_closed_without_partial_mutation() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, 2);

    let mut wrong_schema = explicit_prior();
    wrong_schema.schema_version = ACTION_TRANSITION_PRIOR_SCHEMA_V1 + 1;
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, wrong_schema),
        ActionTransitionPriorError::UnsupportedSchemaVersion { .. }
    ));

    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 2, explicit_prior()),
        ActionTransitionPriorError::ActionOutOfRange { .. }
    ));

    let mut wrong_rows = explicit_prior();
    wrong_rows.transition_matrix.pop();
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, wrong_rows),
        ActionTransitionPriorError::MatrixRowCountMismatch { .. }
    ));

    let mut wrong_columns = explicit_prior();
    wrong_columns.transition_matrix[1].pop();
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, wrong_columns),
        ActionTransitionPriorError::MatrixColumnCountMismatch { .. }
    ));

    let mut non_finite_matrix = explicit_prior();
    non_finite_matrix.transition_matrix[0][0] = f64::NAN;
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, non_finite_matrix),
        ActionTransitionPriorError::NonFiniteMatrixValue { .. }
    ));

    let mut out_of_range = explicit_prior();
    out_of_range.transition_matrix[0][0] = 1.1;
    out_of_range.transition_matrix[0][1] = -0.1;
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, out_of_range),
        ActionTransitionPriorError::ProbabilityOutOfRange { .. }
    ));

    let mut unnormalized = explicit_prior();
    unnormalized.transition_matrix[0][0] = 0.7;
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, unnormalized),
        ActionTransitionPriorError::RowNotNormalized { .. }
    ));

    let mut wrong_bias_length = explicit_prior();
    wrong_bias_length.transition_bias.pop();
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, wrong_bias_length),
        ActionTransitionPriorError::BiasLengthMismatch { .. }
    ));

    let mut non_finite_bias = explicit_prior();
    non_finite_bias.transition_bias[2] = f64::INFINITY;
    assert!(matches!(
        assert_rejected_without_mutation(&mut model, 0, non_finite_bias),
        ActionTransitionPriorError::NonFiniteBiasValue { .. }
    ));
}

#[test]
fn learned_and_explicit_action_identity_survives_json_round_trip() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, 3);
    model
        .apply_action_transition_prior_v1(2, explicit_prior())
        .expect("valid prior should apply");
    model.learn_transition(
        &one_hot_state(0),
        1,
        &one_hot_state(3),
        &observation(),
    );

    let encoded = serde_json::to_string(&model).expect("model should serialize");
    let restored: GenerativeModel =
        serde_json::from_str(&encoded).expect("model should deserialize");

    assert_eq!(restored.transition_matrices, model.transition_matrices);
    assert_eq!(restored.transition_bias, model.transition_bias);
    assert_eq!(restored.num_actions, model.num_actions);
    assert_eq!(restored.state_dim, model.state_dim);

    let state = asymmetric_state();
    for action in 0..model.num_actions {
        assert_eq!(
            restored.predict_next_state(&state, action).mean,
            model.predict_next_state(&state, action).mean
        );
    }
}

#[test]
fn single_state_prior_is_well_formed_and_action_neutral() {
    let model = GenerativeModel::new(1, 1, 3);
    assert_eq!(model.transition_matrices, vec![vec![vec![1.0]]; 3]);
}
