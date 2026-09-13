// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ContinuousHV, HlsConfig, HlsEligibilityTrace, HolographicLiquidCell, step_with_eligibility,
};

#[test]
fn traced_and_untraced_forward_paths_remain_identical() {
    let config = HlsConfig {
        dim: 128,
        state_norm_limit: f32::INFINITY,
        ..HlsConfig::default()
    };
    let mut traced = HolographicLiquidCell::try_new(config.clone(), 42).unwrap();
    let mut ordinary = HolographicLiquidCell::try_new(config, 42).unwrap();
    let initial = ContinuousHV::new_random(128, 43).scale(0.25);
    traced.set_state(initial.clone()).unwrap();
    ordinary.set_state(initial).unwrap();
    let mut eligibility = HlsEligibilityTrace::zeros(128);

    for (step, dt) in [0.001_f32, 0.017, 0.13, 0.8, 4.0].into_iter().enumerate() {
        let input = ContinuousHV::new_random(128, 100 + step as u64).scale(0.2);
        step_with_eligibility(&mut traced, &mut eligibility, dt, &input).unwrap();
        ordinary.step(dt, &input).unwrap();
        assert_eq!(traced.state().values, ordinary.state().values);
    }
    assert_eq!(eligibility.scalar_count(), 6 * 128);
}

#[test]
fn current_loss_gradient_has_same_shape_as_trainable_surface() {
    let config = HlsConfig {
        dim: 64,
        state_norm_limit: f32::INFINITY,
        ..HlsConfig::default()
    };
    let mut cell = HolographicLiquidCell::try_new(config, 7).unwrap();
    let mut trace = HlsEligibilityTrace::zeros(64);
    for step in 0..8_u64 {
        let input = ContinuousHV::new_random(64, 200 + step).scale(0.1);
        step_with_eligibility(&mut cell, &mut trace, 0.03 + step as f32 * 0.01, &input)
            .unwrap();
    }
    let learning_signal = ContinuousHV::new_random(64, 300).scale(0.2);
    let gradient = trace.parameter_gradient(&learning_signal).unwrap();
    assert_eq!(gradient.scalar_count(), cell.parameter_count());
    assert!(gradient.l2_norm().is_finite());
}
