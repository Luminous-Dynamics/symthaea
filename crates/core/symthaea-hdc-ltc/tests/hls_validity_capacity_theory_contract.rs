// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ValidityCapacityAxis, ValidityCapacityNullModel, ValidityCapacityPlan,
};

#[test]
fn frozen_baseline_has_exact_pre_result_variance_prediction() {
    let model = ValidityCapacityNullModel::new(4096, 8, 8, 128).unwrap();

    assert_eq!(model.represented_facts, 1024);
    assert_eq!(model.facts_per_dimension, 0.25);
    assert_eq!(model.same_checkpoint_target_interferers, 7);
    assert_eq!(model.cross_checkpoint_interferers, 1016);

    // target = [7 + (1016 / 2)] / 4096 = 515 / 4096
    assert_eq!(model.target_noise_variance, 515.0 / 4096.0);
    // distractor = [8 + (1016 / 2)] / 4096 = 516 / 4096
    assert_eq!(model.distractor_noise_variance, 516.0 / 4096.0);
    assert_eq!(model.large_horizon_variance_proxy, 0.125);
    assert_eq!(model.target_variance_correction(), 3.0 / 4096.0);
    assert_eq!(model.distractor_variance_correction(), 4.0 / 4096.0);
}

#[test]
fn theory_predictions_are_defined_for_every_frozen_research_case() {
    let plan = ValidityCapacityPlan::research_v0();
    assert_eq!(plan.cases.len(), 27);

    for case in plan.cases {
        let model = ValidityCapacityNullModel::from_case(case).unwrap();
        assert_eq!(model.dim, case.dim);
        assert_eq!(model.key_count, case.key_count);
        assert_eq!(model.candidate_count, case.candidate_count);
        assert_eq!(model.horizon, case.horizon);
        assert!(model.target_noise_variance.is_finite());
        assert!(model.distractor_noise_variance.is_finite());
        assert!(model.target_noise_variance >= 0.0);
        assert!(model.distractor_noise_variance >= model.target_noise_variance);
        assert!(model.gaussian_distractor_extreme_scale.is_finite());
    }
}

#[test]
fn equal_rho_does_not_imply_identical_finite_horizon_prediction() {
    // Both cases have rho = K*H/D = 0.25, but the exact finite-horizon correction
    // retains K/D information. This prevents the public theory from silently
    // collapsing all finite cases onto rho alone.
    let first = ValidityCapacityNullModel::new(4096, 8, 8, 128).unwrap();
    let second = ValidityCapacityNullModel::new(4096, 16, 8, 64).unwrap();
    assert_eq!(first.facts_per_dimension, second.facts_per_dimension);
    assert_ne!(first.target_noise_variance, second.target_noise_variance);
    assert_ne!(first.distractor_noise_variance, second.distractor_noise_variance);
}

#[test]
fn span_length_axis_is_a_model_deviation_test_not_a_theory_input() {
    let plan = ValidityCapacityPlan::research_v0();
    let span_cases = plan
        .cases
        .into_iter()
        .filter(|case| case.axis == ValidityCapacityAxis::SpanLength)
        .collect::<Vec<_>>();
    assert_eq!(span_cases.len(), 7);

    let reference = ValidityCapacityNullModel::from_case(span_cases[0]).unwrap();
    for case in span_cases.into_iter().skip(1) {
        assert_eq!(ValidityCapacityNullModel::from_case(case).unwrap(), reference);
    }
}
