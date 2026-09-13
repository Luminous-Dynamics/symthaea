// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ContextualHolographicLiquidCell, ContinuousHV, HlsActivation, HlsConfig, UnitaryRole,
};

fn config(activation: HlsActivation) -> HlsConfig {
    HlsConfig {
        dim: 256,
        activation,
        ..HlsConfig::default()
    }
}

#[test]
fn contextual_composition_keeps_binding_equivariance_across_sweep() {
    let activations = [
        HlsActivation::Tanh,
        HlsActivation::Identity,
        HlsActivation::BoundedTanh { bound: 0.5 },
    ];
    let dts = [0.0001_f32, 0.01, 0.5, 10.0, 1000.0];

    for activation in activations {
        for seed in 0_u64..8 {
            let mut cell = ContextualHolographicLiquidCell::try_new(
                config(activation),
                vec![1, 7, 31, 127],
                0.75,
                seed,
            )
            .unwrap();
            cell.set_state(ContinuousHV::new_random(256, seed + 1000))
                .unwrap();
            let input = ContinuousHV::new_random(256, seed + 2000);
            let role = UnitaryRole::new(256, seed + 3000);

            for dt in dts {
                let error = cell
                    .binding_equivariance_error(&role, &input, dt)
                    .unwrap();
                assert!(
                    error <= 3e-6,
                    "contextual binding equivariance failed: activation={activation:?} seed={seed} dt={dt} error={error}"
                );
            }
        }
    }
}

#[test]
fn contextual_path_adds_remote_influence_absent_from_diagonal_input_channel() {
    let mut cell = ContextualHolographicLiquidCell::try_new(
        HlsConfig {
            dim: 32,
            ..HlsConfig::default()
        },
        vec![3],
        1.0,
        77,
    )
    .unwrap();

    let mut input = ContinuousHV::new(32);
    input.values[0] = 1.0;
    let baseline_effective = cell.effective_input(&input).unwrap();

    let mut state = ContinuousHV::new(32);
    state.values[3] = 2.0;
    cell.set_state(state).unwrap();
    let contextual_effective = cell.effective_input(&input).unwrap();

    assert_ne!(baseline_effective.values[0], contextual_effective.values[0]);
    assert_eq!(input.values[0], 1.0);
}
