use std::collections::BTreeMap;

use symthaea_fep::efe_diagnostics::{
    EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1, ExpectedFreeEnergyDecompositionV1,
    expected_free_energy_decomposition_v1,
};
use symthaea_fep::efe_policy_probe::{
    DiagnosticPolicyProbeError, diagnostic_policy_probabilities_v1,
};
use symthaea_fep::generative_model::ActionTransitionPriorV1;
use symthaea_fep::{
    ExpectedFreeEnergyComputer, GenerativeModel, HiddenState, ModelConfidenceTracker,
};

const STATE_DIM: usize = 2;
const OBS_DIM: usize = 2;
const ACTIONS: usize = 2;

fn synthetic(action: usize, score: f64) -> ExpectedFreeEnergyDecompositionV1 {
    ExpectedFreeEnergyDecompositionV1 {
        schema_version: EXPECTED_FREE_ENERGY_DIAGNOSTIC_SCHEMA_V1,
        action,
        transition_from_state: 0,
        pragmatic_cost_nats: 0.0,
        hidden_state_information_gain_nats: 0.0,
        transition_parameter_information_gain_nats: 0.0,
        canonical_diagnostic_score_nats: score,
        legacy_exploration_heuristic_unitless: 1.0,
        expected_observation: vec![],
    }
}

fn probabilities_by_action(
    candidates: &[ExpectedFreeEnergyDecompositionV1],
    temperature: f64,
) -> BTreeMap<usize, f64> {
    diagnostic_policy_probabilities_v1(candidates, temperature)
        .expect("policy probe should be valid")
        .actions
        .into_iter()
        .map(|entry| (entry.action, entry.probability))
        .collect()
}

fn state() -> HiddenState {
    let mut state = HiddenState::new(STATE_DIM);
    state.mean = vec![0.7, 0.3];
    state.precision = vec![2.0, 3.0];
    state
}

fn evidence() -> ModelConfidenceTracker {
    ModelConfidenceTracker::new(ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1)
}

fn zero_preference_efe() -> ExpectedFreeEnergyComputer {
    let mut efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    efe.set_preferences_with_precisions(vec![0.0; OBS_DIM], vec![0.0; OBS_DIM]);
    efe
}

fn diagnose_pair(
    model: &GenerativeModel,
    evidence: &ModelConfidenceTracker,
    efe: &ExpectedFreeEnergyComputer,
) -> [ExpectedFreeEnergyDecompositionV1; 2] {
    let current = state();
    [
        expected_free_energy_decomposition_v1(0, 0, &current, model, evidence, efe)
            .expect("action 0 diagnostic should be valid"),
        expected_free_energy_decomposition_v1(1, 0, &current, model, evidence, efe)
            .expect("action 1 diagnostic should be valid"),
    ]
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn softmax_obeys_exact_log_odds_sign_and_temperature() {
    let candidates = [synthetic(10, 0.25), synthetic(20, 1.75)];
    let temperature = 0.5;
    let probabilities = probabilities_by_action(&candidates, temperature);

    assert!(probabilities[&10] > probabilities[&20]);
    let observed_log_odds = (probabilities[&10] / probabilities[&20]).ln();
    let expected_log_odds = -(0.25_f64 - 1.75_f64) / temperature;
    assert_close(observed_log_odds, expected_log_odds, 1e-12);
}

#[test]
fn equal_scores_are_uniform_and_candidate_order_is_irrelevant() {
    let original = [synthetic(3, 0.7), synthetic(8, 0.7), synthetic(13, 0.7)];
    let reversed = [original[2].clone(), original[1].clone(), original[0].clone()];

    let first = probabilities_by_action(&original, 1.0);
    let second = probabilities_by_action(&reversed, 1.0);

    assert_eq!(first, second);
    for probability in first.values() {
        assert_close(*probability, 1.0 / 3.0, 1e-12);
    }
}

#[test]
fn action_identity_permutation_moves_probability_with_score_not_numeric_id() {
    let original = [synthetic(0, -0.5), synthetic(1, 0.5)];
    let permuted = [synthetic(1, -0.5), synthetic(0, 0.5)];

    let first = probabilities_by_action(&original, 1.0);
    let second = probabilities_by_action(&permuted, 1.0);

    assert_close(first[&0], second[&1], 1e-12);
    assert_close(first[&1], second[&0], 1e-12);
}

#[test]
fn legacy_heuristic_cannot_change_probe_probabilities() {
    let original = [synthetic(0, -0.25), synthetic(1, 0.25)];
    let mut changed = original.clone();
    changed[0].legacy_exploration_heuristic_unitless = 1e-12;
    changed[1].legacy_exploration_heuristic_unitless = 1e12;

    assert_eq!(
        probabilities_by_action(&original, 1.0),
        probabilities_by_action(&changed, 1.0)
    );
}

#[test]
fn malformed_probe_inputs_fail_closed() {
    assert!(matches!(
        diagnostic_policy_probabilities_v1(&[], 1.0),
        Err(DiagnosticPolicyProbeError::EmptyCandidateSet)
    ));

    let one = [synthetic(0, 0.0)];
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            diagnostic_policy_probabilities_v1(&one, invalid),
            Err(DiagnosticPolicyProbeError::InvalidTemperature { .. })
        ));
    }

    let duplicate = [synthetic(0, 0.0), synthetic(0, 1.0)];
    assert!(matches!(
        diagnostic_policy_probabilities_v1(&duplicate, 1.0),
        Err(DiagnosticPolicyProbeError::DuplicateAction { action: 0 })
    ));

    let mut wrong_schema = synthetic(0, 0.0);
    wrong_schema.schema_version += 1;
    assert!(matches!(
        diagnostic_policy_probabilities_v1(&[wrong_schema], 1.0),
        Err(DiagnosticPolicyProbeError::UnsupportedDiagnosticSchema { .. })
    ));

    let mut non_finite = synthetic(0, 0.0);
    non_finite.canonical_diagnostic_score_nats = f64::NAN;
    assert!(matches!(
        diagnostic_policy_probabilities_v1(&[non_finite], 1.0),
        Err(DiagnosticPolicyProbeError::NonFiniteCanonicalScore { .. })
    ));
}

#[test]
fn parameter_information_advantage_diminishes_after_matched_evidence() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let mut evidence = evidence();
    let efe = zero_preference_efe();

    for _ in 0..64 {
        evidence.update_transition(0, 0, 0);
        evidence.update_transition(0, 0, 1);
    }

    let before = diagnose_pair(&model, &evidence, &efe);
    assert_close(
        before[0].hidden_state_information_gain_nats,
        before[1].hidden_state_information_gain_nats,
        1e-12,
    );
    assert!(
        before[1].transition_parameter_information_gain_nats
            > before[0].transition_parameter_information_gain_nats
    );
    let before_probabilities = probabilities_by_action(&before, 1.0);
    assert!(before_probabilities[&1] > before_probabilities[&0]);

    for _ in 0..64 {
        evidence.update_transition(1, 0, 0);
        evidence.update_transition(1, 0, 1);
    }

    let after = diagnose_pair(&model, &evidence, &efe);
    assert_close(
        after[0].transition_parameter_information_gain_nats,
        after[1].transition_parameter_information_gain_nats,
        1e-12,
    );
    assert_close(
        after[0].canonical_diagnostic_score_nats,
        after[1].canonical_diagnostic_score_nats,
        1e-12,
    );
    let after_probabilities = probabilities_by_action(&after, 1.0);
    assert_close(after_probabilities[&0], 0.5, 1e-12);
    assert_close(after_probabilities[&1], 0.5, 1e-12);
}

#[test]
fn salience_isolation_moves_probability_toward_more_informative_action() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let efe = zero_preference_efe();

    model
        .apply_action_transition_prior_v1(
            0,
            ActionTransitionPriorV1::new(
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![0.0, 0.0],
            ),
        )
        .unwrap();
    model
        .apply_action_transition_prior_v1(
            1,
            ActionTransitionPriorV1::new(
                vec![vec![0.5, 0.5], vec![0.5, 0.5]],
                vec![0.0, 0.0],
            ),
        )
        .unwrap();

    let diagnostics = diagnose_pair(&model, &evidence, &efe);
    assert_close(
        diagnostics[0].transition_parameter_information_gain_nats,
        diagnostics[1].transition_parameter_information_gain_nats,
        1e-12,
    );
    let probabilities = probabilities_by_action(&diagnostics, 1.0);

    let more_informative = if diagnostics[0].hidden_state_information_gain_nats
        > diagnostics[1].hidden_state_information_gain_nats
    {
        0
    } else {
        1
    };
    let less_informative = 1 - more_informative;
    assert!(probabilities[&more_informative] > probabilities[&less_informative]);
}

#[test]
fn pragmatic_isolation_moves_probability_toward_preference_consistent_action() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let current = state();
    let shared_transition = model.transition_matrices[0].clone();

    model
        .apply_action_transition_prior_v1(
            0,
            ActionTransitionPriorV1::new(shared_transition.clone(), vec![0.0, 0.0]),
        )
        .unwrap();
    model
        .apply_action_transition_prior_v1(
            1,
            ActionTransitionPriorV1::new(shared_transition, vec![0.25, -0.15]),
        )
        .unwrap();

    let target = model.predict_observation(&model.predict_next_state(&current, 0));
    let mut efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    efe.set_preferences(target, 4.0);

    let diagnostics = [
        expected_free_energy_decomposition_v1(0, 0, &current, &model, &evidence, &efe).unwrap(),
        expected_free_energy_decomposition_v1(1, 0, &current, &model, &evidence, &efe).unwrap(),
    ];
    assert_close(
        diagnostics[0].hidden_state_information_gain_nats,
        diagnostics[1].hidden_state_information_gain_nats,
        1e-12,
    );
    assert_close(
        diagnostics[0].transition_parameter_information_gain_nats,
        diagnostics[1].transition_parameter_information_gain_nats,
        1e-12,
    );
    assert!(diagnostics[0].pragmatic_cost_nats < diagnostics[1].pragmatic_cost_nats);

    let probabilities = probabilities_by_action(&diagnostics, 1.0);
    assert!(probabilities[&0] > probabilities[&1]);
}
