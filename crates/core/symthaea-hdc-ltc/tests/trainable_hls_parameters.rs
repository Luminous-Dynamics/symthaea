// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ContinuousHV, HlsConfig, HlsError, HlsParameters, HolographicLiquidCell, UnitaryRole,
};

#[test]
fn public_parameter_snapshot_roundtrips_exactly() {
    let mut cell = HolographicLiquidCell::try_new(
        HlsConfig {
            dim: 256,
            ..HlsConfig::default()
        },
        42,
    )
    .unwrap();

    let original = cell.parameters();
    assert_eq!(cell.parameter_count(), 6 * 256);
    assert_eq!(original.scalar_count(), cell.parameter_count());

    let mut modified = original.clone();
    for value in &mut modified.recurrent_weight.values {
        *value *= 0.5;
    }
    for value in &mut modified.tau_state_weight.values {
        *value *= -0.75;
    }
    cell.set_parameters(modified.clone()).unwrap();
    assert_eq!(cell.parameters(), modified);
}

#[test]
fn invalid_public_parameter_snapshot_is_fail_closed() {
    let mut cell = HolographicLiquidCell::try_new(
        HlsConfig {
            dim: 128,
            ..HlsConfig::default()
        },
        7,
    )
    .unwrap();
    let before = cell.parameters();

    let mut invalid = before.clone();
    invalid.gate_bias.values[3] = f32::NAN;
    assert!(matches!(
        cell.set_parameters(invalid),
        Err(HlsError::NonFiniteParameter("gate_bias"))
    ));
    assert_eq!(cell.parameters(), before);
}

#[test]
fn optimizer_delta_does_not_weaken_binding_equivariance() {
    let mut cell = HolographicLiquidCell::try_new(
        HlsConfig {
            dim: 256,
            ..HlsConfig::default()
        },
        11,
    )
    .unwrap();

    let mut delta = HlsParameters::zeros(256);
    for (i, value) in delta.recurrent_weight.values.iter_mut().enumerate() {
        *value = ((i % 9) as f32 - 4.0) * 0.3;
    }
    for (i, value) in delta.input_weight.values.iter_mut().enumerate() {
        *value = ((i % 7) as f32 - 3.0) * 0.2;
    }
    for (i, value) in delta.tau_state_weight.values.iter_mut().enumerate() {
        *value = ((i % 5) as f32 - 2.0) * 0.4;
    }
    for (i, value) in delta.gate_state_weight.values.iter_mut().enumerate() {
        *value = ((i % 11) as f32 - 5.0) * 0.1;
    }
    for (i, value) in delta.gate_input_weight.values.iter_mut().enumerate() {
        *value = ((i % 13) as f32 - 6.0) * 0.08;
    }
    for (i, value) in delta.gate_bias.values.iter_mut().enumerate() {
        *value = ((i % 3) as f32 - 1.0) * 0.2;
    }

    cell.apply_parameter_delta(&delta, 0.25, 2.0).unwrap();
    cell.set_state(ContinuousHV::new_random(256, 12)).unwrap();
    let input = ContinuousHV::new_random(256, 13);
    let role = UnitaryRole::new(256, 14);
    let error = cell
        .binding_equivariance_error(&role, &input, 0.071)
        .unwrap();
    assert!(error <= 1e-6, "post-update equivariance error={error}");
}
