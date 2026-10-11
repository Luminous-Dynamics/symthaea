// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Regression fixtures for TEMPORAL-STATE-000 / #3587.
//!
//! These tests deliberately exercise the current public classic-CfC lifecycle
//! surface without going through `TemporalNetwork`. They freeze two facts that
//! the temporal-state repair must preserve as historical evidence:
//!
//! 1. the ordinary classic-CfC cognitive-loop shape is multilayer by default;
//! 2. `read_state()` followed by `inject()` is not an exact restore for a
//!    multilayer network because the projected final-layer state is replicated
//!    into every layer.
//!
//! They also prove the proposed P0 containment mechanism: probing a detached
//! clone cannot mutate the live network's evolution state.

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};

fn small_two_layer_config() -> CfCNetworkConfig {
    CfCNetworkConfig {
        input_dim: 4,
        hidden_dim: 4,
        num_layers: 2,
        output_dim: 2,
        ..CfCNetworkConfig::default()
    }
}

#[test]
fn ordinary_classic_cfc_construction_is_multilayer() {
    assert_eq!(CfCNetworkConfig::default().num_layers, 2);

    let network = CfCNetwork::new_with_input(4, 4);
    assert_eq!(network.config().num_layers, 2);
}

#[test]
fn projected_read_then_inject_collapses_distinct_layer_state() {
    let mut network = CfCNetwork::new(small_two_layer_config());

    let fast_layer = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let slow_layer = Array1::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
    network.set_state(vec![fast_layer.clone(), slow_layer.clone()]);

    let before = network.state();
    assert_eq!(before, vec![fast_layer.clone(), slow_layer.clone()]);

    // Legacy projected observation: only the final cell is returned.
    let projected = network.read_state().expect("read projected CfC state");
    assert_eq!(projected, slow_layer);

    // Legacy injection is therefore not an inverse of read_state() for a
    // multilayer network: it writes that one projected vector into every cell.
    network
        .inject(&projected)
        .expect("inject projected CfC state");

    let after = network.state();
    assert_eq!(after.len(), 2);
    assert_eq!(after[0], projected);
    assert_eq!(after[1], projected);
    assert_ne!(after, before);
    assert_ne!(after[0], fast_layer);
}

#[test]
fn detached_clone_prediction_preserves_live_multilayer_state() {
    let mut live = CfCNetwork::new(small_two_layer_config());
    live.set_state(vec![
        Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]),
        Array1::from_vec(vec![0.5, 0.6, 0.7, 0.8]),
    ]);

    let before = live.state();
    let mut probe = live.clone();
    let input = Array1::from_vec(vec![0.9, -0.4, 0.2, 0.7]);

    let prediction = probe
        .predict_forward(&input, 0.25)
        .expect("detached CfC prediction");

    assert_eq!(prediction.len(), 2);
    assert_eq!(live.state(), before);
}
