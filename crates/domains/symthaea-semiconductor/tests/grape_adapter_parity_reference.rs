use symthaea_semiconductor::benchmark::{
    GrapeAdapterKindV1, GrapeAdapterPolicyV1, GrapeAdapterRuntimeV1, GrapeHierarchicalContextStateV1, GrapeRuntimePolicyV1, create_grape_feature_adapter_v1,
};

fn policy(kind: GrapeAdapterKindV1) -> GrapeAdapterPolicyV1 {
    GrapeAdapterPolicyV1 {
        schema: "luminous.semi-bench.grape-adapter-parity.v1".into(),
        adapter_kind: kind,
        nlms_learning_rate: 0.01,
        rls_forgetting_factor: 0.99,
        rls_ridge_initial: 1.0,
        nlms_norm_floor: 1.0e-3,
        positive_definite_tolerance: 1.0e-12,
    }
}

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:.16e}, expected={expected:.16e}"
    );
}

#[test]
fn independent_two_dimensional_rls_reference_vector() {
    let policy = policy(GrapeAdapterKindV1::Rls);
    let mut state = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Rls, 2, &policy).unwrap();

    // Frozen hand calculation: P=I, x=[1,2], lambda=.99, error=.5.
    // denominator=5.99, k=[1/5.99, 2/5.99].
    state.update(&policy, &[1.0, 2.0], 0.5).unwrap();

    let denom = 5.99_f64;
    let gain_0 = 1.0 / denom;
    let gain_1 = 2.0 / denom;
    let expected_p00 = (1.0 - gain_0) / 0.99;
    let expected_p01 = (-2.0 / denom) / 0.99;
    let expected_p11 = (1.0 - 4.0 / denom) / 0.99;

    close(state.weights[0], 0.5 * gain_0, 1.0e-14, "RLS weight 0");
    close(state.weights[1], 0.5 * gain_1, 1.0e-14, "RLS weight 1");
    let covariance = state.covariance.as_ref().unwrap();
    close(covariance[0][0], expected_p00, 1.0e-13, "RLS P00");
    close(covariance[0][1], expected_p01, 1.0e-13, "RLS P01");
    close(covariance[1][0], expected_p01, 1.0e-13, "RLS P10");
    close(covariance[1][1], expected_p11, 1.0e-13, "RLS P11");
}

#[test]
fn independent_nlms_reference_vector() {
    let policy = policy(GrapeAdapterKindV1::Nlms);
    let mut state = create_grape_feature_adapter_v1(GrapeAdapterKindV1::Nlms, 2, &policy).unwrap();

    // The pinned GRAPE source uses ||x||^2 + 1e-3, not a unit-normalized update.
    state.update(&policy, &[1.0, 2.0], 0.5).unwrap();
    close(state.weights[0], 0.005 / 5.001, 1.0e-14, "NLMS weight 0");
    close(state.weights[1], 0.010 / 5.001, 1.0e-14, "NLMS weight 1");
}

#[test]
fn independent_train_predict_update_reference_vector() {
    let runtime_policy = GrapeRuntimePolicyV1 {
        adapter_policy: policy(GrapeAdapterKindV1::Nlms),
        adapter_feature_count: 1,
        context_ema_alpha: 0.05,
        context_minimum_support: 10,
        temporal_ema_alpha: 0.5,
        temporal_activation_percentile: 96.0,
    };
    let training_features = vec![
        vec![1.0, 5.0],
        vec![2.0, 5.0],
        vec![3.0, 5.0],
        vec![4.0, 5.0],
    ];
    let training_targets = vec![1.0, 2.0, 3.0, 4.0];
    let global_predictions = vec![0.0, 0.0, 0.0, 0.0];

    let mut runtime = GrapeAdapterRuntimeV1::initialize(
        runtime_policy,
        &training_features,
        &training_targets,
        &global_predictions,
    )
    .unwrap();

    // Residual correlation selects feature 0; feature 1 is constant.
    assert_eq!(runtime.selected_feature_indices, vec![0]);
    let prediction = runtime.predict(2.0, &[2.0, 5.0], Some("eqpA_ch1")).unwrap();
    close(prediction.adapter_correction, 0.0, 1.0e-14, "initial adapter");
    close(prediction.context_correction, 0.0, 1.0e-14, "initial context");
    close(prediction.final_prediction, 2.0, 1.0e-14, "prediction");

    let receipt = runtime.update_from_prediction(&prediction, 3.0).unwrap();
    let expected_weight = 0.01 * 1.0 * 2.0 / (4.0 + 1.0e-3);
    let expected_adapter = expected_weight * 2.0;
    let expected_context_residual = 3.0 - (2.0 + expected_adapter);
    close(runtime.state.adapter.weights[0], expected_weight, 1.0e-14, "runtime NLMS weight");
    close(runtime.state.temporal.residual_ema, 0.5, 1.0e-14, "temporal EMA");
    close(runtime.state.context.global_mean, 0.05 * expected_context_residual, 1.0e-14, "global context EMA");
    close(receipt.feature_error, 1.0, 1.0e-14, "feature error");
    close(receipt.temporal_error, 1.0, 1.0e-14, "temporal error");
    close(
        receipt.context_residual.unwrap(),
        expected_context_residual,
        1.0e-14,
        "context residual"
    );
    assert_eq!(runtime.state.context.equipment["eqpA"].count, 1);
    assert_eq!(runtime.state.context.chamber["ch1"].count, 1);
    assert_eq!(runtime.state.context.equipment_chamber["eqpA_ch1"].count, 1);
    assert_eq!(runtime.state.accepted_update_count, 1);
}

#[test]
fn context_reliability_reference_vector_is_hand_computable() {
    let mut context = GrapeHierarchicalContextStateV1::new(0.5, 2).unwrap();
    context.update(Some("eqpA_ch1"), 2.0).unwrap();
    context.update(Some("eqpA_ch2"), 6.0).unwrap();
    context.update(Some("eqpB_ch1"), 10.0).unwrap();

    let global_weight = 3.0 / 5.0;
    let equipment_weight = 2.0 / 4.0;
    let chamber_weight = 2.0 / 4.0;
    let pair_weight = 1.0 / 3.0;
    let expected = (
        context.global_mean * global_weight
            + context.equipment["eqpA"].mean * equipment_weight
            + context.chamber["ch1"].mean * chamber_weight
            + context.equipment_chamber["eqpA_ch1"].mean * pair_weight
    ) / (global_weight + equipment_weight + chamber_weight + pair_weight);
    close(
        context.correction(Some("eqpA_ch1")).unwrap(),
        expected,
        1.0e-14,
        "four-level context correction"
    );
}
