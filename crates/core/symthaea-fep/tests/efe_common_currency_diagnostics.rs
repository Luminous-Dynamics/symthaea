use symthaea_fep::efe_diagnostics::{
    ExpectedFreeEnergyDiagnosticError, expected_free_energy_decomposition_v1,
};
use symthaea_fep::generative_model::ActionTransitionPriorV1;
use symthaea_fep::{
    ExpectedFreeEnergyComputer, GenerativeModel, HiddenState, ModelConfidenceTracker,
};

const STATE_DIM: usize = 2;
const OBS_DIM: usize = 2;
const ACTIONS: usize = 2;

fn state() -> HiddenState {
    let mut state = HiddenState::new(STATE_DIM);
    state.mean = vec![0.7, 0.3];
    state.precision = vec![2.0, 3.0];
    state
}

fn evidence() -> ModelConfidenceTracker {
    ModelConfidenceTracker::new(ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1)
}

fn efe() -> ExpectedFreeEnergyComputer {
    let mut efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    efe.set_preferences(vec![0.6, 0.4], 2.0);
    efe
}

fn diagnose(
    action: usize,
    model: &GenerativeModel,
    evidence: &ModelConfidenceTracker,
    efe: &ExpectedFreeEnergyComputer,
) -> symthaea_fep::ExpectedFreeEnergyDecompositionV1 {
    expected_free_energy_decomposition_v1(action, 0, &state(), model, evidence, efe)
        .expect("diagnostic should be valid")
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "left={left:.16e}, right={right:.16e}, tolerance={tolerance:.3e}"
    );
}

#[test]
fn parameter_information_changes_only_parameter_channel_under_matched_model() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let mut evidence = evidence();
    let efe = efe();

    for _ in 0..64 {
        evidence.update_transition(0, 0, 0);
        evidence.update_transition(0, 0, 1);
    }

    let learned = diagnose(0, &model, &evidence, &efe);
    let unseen = diagnose(1, &model, &evidence, &efe);

    assert_close(learned.pragmatic_cost_nats, unseen.pragmatic_cost_nats, 1e-12);
    assert_close(
        learned.hidden_state_information_gain_nats,
        unseen.hidden_state_information_gain_nats,
        1e-12,
    );
    assert!(
        unseen.transition_parameter_information_gain_nats
            > learned.transition_parameter_information_gain_nats,
        "less-observed transition dynamics should offer more parameter information"
    );
    assert!(
        unseen.canonical_diagnostic_score_nats < learned.canonical_diagnostic_score_nats,
        "positive information gain must lower the minimization score"
    );
}

#[test]
fn hidden_state_salience_changes_without_parameter_or_pragmatic_leakage() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let mut efe = efe();
    efe.set_preferences_with_precisions(vec![0.0, 0.0], vec![0.0, 0.0]);

    let identity = ActionTransitionPriorV1::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
    );
    let averaging = ActionTransitionPriorV1::new(
        vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        vec![0.0, 0.0],
    );
    model
        .apply_action_transition_prior_v1(0, identity)
        .expect("identity prior should apply");
    model
        .apply_action_transition_prior_v1(1, averaging)
        .expect("averaging prior should apply");

    let first = diagnose(0, &model, &evidence, &efe);
    let second = diagnose(1, &model, &evidence, &efe);

    assert_eq!(first.pragmatic_cost_nats, 0.0);
    assert_eq!(second.pragmatic_cost_nats, 0.0);
    assert_close(
        first.transition_parameter_information_gain_nats,
        second.transition_parameter_information_gain_nats,
        1e-12,
    );
    assert!(
        (first.hidden_state_information_gain_nats - second.hidden_state_information_gain_nats)
            .abs()
            > 1e-8,
        "different action-conditioned covariance should be visible in salience"
    );

    let higher_information = first
        .hidden_state_information_gain_nats
        .max(second.hidden_state_information_gain_nats);
    let lower_score = first
        .canonical_diagnostic_score_nats
        .min(second.canonical_diagnostic_score_nats);
    assert_close(lower_score, -higher_information - first.transition_parameter_information_gain_nats, 1e-12);
}

#[test]
fn pragmatic_cost_changes_under_mean_shift_without_salience_or_parameter_leakage() {
    let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();

    let shared_transition = model.transition_matrices[0].clone();
    model
        .apply_action_transition_prior_v1(
            0,
            ActionTransitionPriorV1::new(shared_transition.clone(), vec![0.0, 0.0]),
        )
        .expect("baseline prior should apply");
    model
        .apply_action_transition_prior_v1(
            1,
            ActionTransitionPriorV1::new(shared_transition, vec![0.25, -0.15]),
        )
        .expect("biased prior should apply");

    let current = state();
    let target = model.predict_observation(&model.predict_next_state(&current, 0));
    let mut efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    efe.set_preferences(target, 4.0);

    let baseline = expected_free_energy_decomposition_v1(0, 0, &current, &model, &evidence, &efe)
        .expect("baseline diagnostic should be valid");
    let shifted = expected_free_energy_decomposition_v1(1, 0, &current, &model, &evidence, &efe)
        .expect("shifted diagnostic should be valid");

    assert_close(
        baseline.hidden_state_information_gain_nats,
        shifted.hidden_state_information_gain_nats,
        1e-12,
    );
    assert_close(
        baseline.transition_parameter_information_gain_nats,
        shifted.transition_parameter_information_gain_nats,
        1e-12,
    );
    assert_close(baseline.pragmatic_cost_nats, 0.0, 1e-15);
    assert!(shifted.pragmatic_cost_nats > baseline.pragmatic_cost_nats);
    assert!(
        shifted.canonical_diagnostic_score_nats > baseline.canonical_diagnostic_score_nats,
        "greater pragmatic cost should raise a minimization score under matched information terms"
    );
}

#[test]
fn legacy_frequency_history_is_explicitly_excluded_from_canonical_score() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let mut efe = efe();

    let before = diagnose(0, &model, &evidence, &efe);
    for _ in 0..20 {
        efe.record_committed_action(0);
    }
    let after = diagnose(0, &model, &evidence, &efe);

    assert_close(before.pragmatic_cost_nats, after.pragmatic_cost_nats, 0.0);
    assert_close(
        before.hidden_state_information_gain_nats,
        after.hidden_state_information_gain_nats,
        0.0,
    );
    assert_close(
        before.transition_parameter_information_gain_nats,
        after.transition_parameter_information_gain_nats,
        0.0,
    );
    assert_close(
        before.canonical_diagnostic_score_nats,
        after.canonical_diagnostic_score_nats,
        0.0,
    );
    assert!(
        after.legacy_exploration_heuristic_unitless
            < before.legacy_exploration_heuristic_unitless
    );
}

#[test]
fn zero_preference_precision_means_no_preference_on_that_dimension() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let current = state();
    let predicted = model.predict_observation(&model.predict_next_state(&current, 0));

    let mut baseline_efe = ExpectedFreeEnergyComputer::new(OBS_DIM);
    baseline_efe.set_preferences_with_precisions(predicted.clone(), vec![0.0, 2.0]);
    let baseline = expected_free_energy_decomposition_v1(
        0,
        0,
        &current,
        &model,
        &evidence,
        &baseline_efe,
    )
    .expect("baseline diagnostic should be valid");

    let mut changed_efe = baseline_efe.clone();
    changed_efe.preferences[0] += 1_000.0;
    let changed = expected_free_energy_decomposition_v1(
        0,
        0,
        &current,
        &model,
        &evidence,
        &changed_efe,
    )
    .expect("zero-precision preference should remain valid");

    assert_close(baseline.pragmatic_cost_nats, changed.pragmatic_cost_nats, 0.0);
    assert_close(
        baseline.canonical_diagnostic_score_nats,
        changed.canonical_diagnostic_score_nats,
        0.0,
    );
}

#[test]
fn malformed_preference_and_evidence_identity_fail_closed() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let current = state();

    let mut wrong_preferences = ExpectedFreeEnergyComputer::new(OBS_DIM);
    wrong_preferences.preferences.pop();
    assert!(matches!(
        expected_free_energy_decomposition_v1(
            0,
            0,
            &current,
            &model,
            &evidence,
            &wrong_preferences,
        ),
        Err(ExpectedFreeEnergyDiagnosticError::PreferenceLengthMismatch { .. })
    ));

    let mut wrong_overrides = efe();
    wrong_overrides.precision_overrides = Some(vec![1.0]);
    assert!(matches!(
        expected_free_energy_decomposition_v1(
            0,
            0,
            &current,
            &model,
            &evidence,
            &wrong_overrides,
        ),
        Err(ExpectedFreeEnergyDiagnosticError::PrecisionOverrideLengthMismatch { .. })
    ));

    let mut invalid_precision = efe();
    invalid_precision.preference_precision = f64::NAN;
    assert!(matches!(
        expected_free_energy_decomposition_v1(
            0,
            0,
            &current,
            &model,
            &evidence,
            &invalid_precision,
        ),
        Err(ExpectedFreeEnergyDiagnosticError::InvalidPreferencePrecision { .. })
    ));

    let wrong_evidence = ModelConfidenceTracker::new(1, STATE_DIM, OBS_DIM, 0.99, 0.1);
    assert!(matches!(
        expected_free_energy_decomposition_v1(
            0,
            0,
            &current,
            &model,
            &wrong_evidence,
            &efe(),
        ),
        Err(ExpectedFreeEnergyDiagnosticError::TransitionEvidenceActionCountMismatch { .. })
    ));

    assert!(matches!(
        expected_free_energy_decomposition_v1(
            ACTIONS,
            0,
            &current,
            &model,
            &evidence,
            &efe(),
        ),
        Err(ExpectedFreeEnergyDiagnosticError::ActionOutOfRange { .. })
    ));
    assert!(matches!(
        expected_free_energy_decomposition_v1(
            0,
            STATE_DIM,
            &current,
            &model,
            &evidence,
            &efe(),
        ),
        Err(ExpectedFreeEnergyDiagnosticError::TransitionFromStateOutOfRange { .. })
    ));
}

#[test]
fn diagnostic_query_is_referentially_pure() {
    let model = GenerativeModel::new(STATE_DIM, OBS_DIM, ACTIONS);
    let evidence = evidence();
    let efe = efe();
    let current = state();

    let model_before = serde_json::to_string(&model).expect("model should serialize");
    let evidence_before = serde_json::to_string(&evidence).expect("evidence should serialize");
    let efe_before = serde_json::to_string(&efe).expect("EFE state should serialize");
    let state_before = serde_json::to_string(&current).expect("state should serialize");

    let first = expected_free_energy_decomposition_v1(0, 0, &current, &model, &evidence, &efe)
        .expect("diagnostic should succeed");
    let second = expected_free_energy_decomposition_v1(0, 0, &current, &model, &evidence, &efe)
        .expect("repeated diagnostic should succeed");

    assert_eq!(first, second);
    assert_eq!(serde_json::to_string(&model).unwrap(), model_before);
    assert_eq!(serde_json::to_string(&evidence).unwrap(), evidence_before);
    assert_eq!(serde_json::to_string(&efe).unwrap(), efe_before);
    assert_eq!(serde_json::to_string(&current).unwrap(), state_before);
}
