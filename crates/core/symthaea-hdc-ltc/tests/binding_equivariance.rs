// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! External theorem tests for Holographic Liquid State binding equivariance.
//!
//! These tests deliberately sweep seeds, irregular `dt`, and every activation
//! admitted by the theorem-bearing cell. A failure means the implementation no
//! longer realizes the claimed algebraic construction.

use symthaea_hdc_ltc::{
    ContinuousHV, HlsActivation, HlsConfig, HlsError, HolographicLiquidCell, UnitaryRole,
};

fn config(activation: HlsActivation) -> HlsConfig {
    HlsConfig {
        dim: 256,
        activation,
        ..HlsConfig::default()
    }
}

#[test]
fn binding_equivariance_survives_seed_dt_and_activation_sweep() {
    let activations = [
        HlsActivation::Tanh,
        HlsActivation::Identity,
        HlsActivation::BoundedTanh { bound: 0.75 },
    ];
    let dts = [1e-4_f32, 0.003, 0.1, 1.0, 17.0, 1000.0];

    for activation in activations {
        for seed in 0_u64..8 {
            let mut cell = HolographicLiquidCell::try_new(config(activation), seed).unwrap();
            cell.set_state(ContinuousHV::new_random(256, seed + 1000))
                .unwrap();
            let input = ContinuousHV::new_random(256, seed + 2000);
            let role = UnitaryRole::new(256, seed + 3000);

            for dt in dts {
                let error = cell
                    .binding_equivariance_error(&role, &input, dt)
                    .unwrap();
                assert!(
                    error <= 2e-6,
                    "binding equivariance violated: activation={activation:?} seed={seed} dt={dt} error={error}"
                );
            }
        }
    }
}

#[test]
fn composed_role_is_equivalent_to_two_role_transforms() {
    let mut cell = HolographicLiquidCell::try_new(config(HlsActivation::Tanh), 42).unwrap();
    cell.set_state(ContinuousHV::new_random(256, 43)).unwrap();
    let input = ContinuousHV::new_random(256, 44);
    let r1 = UnitaryRole::new(256, 45);
    let r2 = UnitaryRole::new(256, 46);
    let composed = r1.compose(&r2);

    let composed_error = cell
        .binding_equivariance_error(&composed, &input, 0.271)
        .unwrap();
    assert!(composed_error <= 2e-6);
}

#[test]
fn theorem_api_fails_closed_on_dimension_mismatch() {
    let cell = HolographicLiquidCell::try_new(config(HlsActivation::Tanh), 1).unwrap();
    let input = ContinuousHV::new_random(256, 2);
    let wrong_role = UnitaryRole::new(128, 3);

    assert_eq!(
        cell.binding_equivariance_error(&wrong_role, &input, 0.1),
        Err(HlsError::DimensionMismatch {
            expected: 256,
            actual: 128,
        })
    );
}
