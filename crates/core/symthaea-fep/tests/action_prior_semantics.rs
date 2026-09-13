//! Regression theorems for generic FEP action-model prior semantics.
//!
//! The generic core has no domain knowledge about integer action identifiers.
//! Therefore construction must be permutation-neutral: initial equality means
//! epistemic ignorance, not an even/odd or ordinal action ontology. Confirmed
//! transition learning may then differentiate individual action models.

use symthaea_fep::generative_model::GenerativeModel;
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
fn single_state_prior_is_well_formed_and_action_neutral() {
    let model = GenerativeModel::new(1, 1, 3);
    assert_eq!(model.transition_matrices, vec![vec![vec![1.0]]; 3]);
}
