use symthaea_fep::efe_diagnostics::{
    ExpectedFreeEnergyDiagnosticError, expected_free_energy_decomposition_v1,
};
use symthaea_fep::{
    ExpectedFreeEnergyComputer, GenerativeModel, HiddenState, ModelConfidenceTracker,
};

#[test]
fn pragmatic_relative_nll_uses_half_precision_squared_error_in_nats() {
    let model = GenerativeModel::new(2, 2, 1);
    let evidence = ModelConfidenceTracker::new(1, 2, 2, 0.99, 0.1);
    let state = HiddenState::new(2);
    let predicted = model.predict_observation(&model.predict_next_state(&state, 0));

    let mut efe = ExpectedFreeEnergyComputer::new(2);
    efe.set_preferences_with_precisions(
        vec![predicted[0] + 1.0, predicted[1]],
        vec![2.0, 0.0],
    );

    let diagnostic = expected_free_energy_decomposition_v1(
        0,
        0,
        &state,
        &model,
        &evidence,
        &efe,
    )
    .expect("diagnostic should be valid");

    // 0.5 * precision(2) * error(1)^2 = 1 nat of relative preference cost.
    assert!((diagnostic.pragmatic_cost_nats - 1.0).abs() < 1e-12);
}

#[test]
fn negative_or_non_finite_preference_precision_fails_closed() {
    let model = GenerativeModel::new(2, 2, 1);
    let evidence = ModelConfidenceTracker::new(1, 2, 2, 0.99, 0.1);
    let state = HiddenState::new(2);

    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        let mut efe = ExpectedFreeEnergyComputer::new(2);
        efe.preference_precision = invalid;
        assert!(matches!(
            expected_free_energy_decomposition_v1(
                0,
                0,
                &state,
                &model,
                &evidence,
                &efe,
            ),
            Err(ExpectedFreeEnergyDiagnosticError::InvalidPreferencePrecision { .. })
        ));
    }
}
