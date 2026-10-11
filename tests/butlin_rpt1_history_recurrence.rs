// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Direct component-level RPT-1 recurrence theorem.
//!
//! The existing Butlin RPT-1 probe measures separation between output
//! centroids for different *current* inputs. That is useful input
//! discrimination evidence, but a feed-forward network can satisfy it.
//!
//! This test asks the recurrence-specific question instead:
//!
//!     history A -> common probe X
//!     history B -> common probe X
//!
//! If the production CfC hidden state carries temporal context, the state
//! after the identical probe X should depend on prior history. If recurrent
//! state is reset immediately before X, that history dependence should
//! collapse while weights and the current input remain identical.
//!
//! This is a component-level theorem only. It does not establish that
//! recurrence is functionally necessary for a downstream cognitive task and
//! does not promote a Butlin support tier.

use ndarray::Array1;
use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};
use symthaea_core::genesis::GenesisSeed;

const HISTORY_STEPS: usize = 12;
const DT: f32 = 0.02;

fn deterministic_network() -> CfCNetwork {
    let mut config = CfCNetworkConfig::default();
    config.enable_online_learning = false;
    let genesis = GenesisSeed::from_phrase("butlin-rpt1-history-recurrence-v1");
    CfCNetwork::from_genesis(config, &genesis, "rpt1-history-test")
}

fn history_a(dim: usize) -> Array1<f32> {
    Array1::from_iter((0..dim).map(|i| if i % 2 == 0 { 0.75 } else { -0.25 }))
}

fn history_b(dim: usize) -> Array1<f32> {
    Array1::from_iter((0..dim).map(|i| if i % 2 == 0 { -0.25 } else { 0.75 }))
}

fn common_probe(dim: usize) -> Array1<f32> {
    Array1::from_iter((0..dim).map(|i| match i % 4 {
        0 => 0.15,
        1 => -0.10,
        2 => 0.05,
        _ => -0.20,
    }))
}

fn state_after_history(history: &Array1<f32>, reset_before_probe: bool) -> Vec<f32> {
    let mut network = deterministic_network();
    for _ in 0..HISTORY_STEPS {
        network.step(history, DT).expect("history step must execute");
    }

    if reset_before_probe {
        network.reset();
    }

    let probe = common_probe(history.len());
    network.step(&probe, DT).expect("probe step must execute");
    network
        .read_state()
        .expect("production CfC state must be readable")
        .to_vec()
}

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn same_current_input_retains_different_prior_histories() {
    let dim = CfCNetworkConfig::default().input_dim;
    let a = state_after_history(&history_a(dim), false);
    let b = state_after_history(&history_b(dim), false);

    assert!(a.iter().all(|x| x.is_finite()));
    assert!(b.iter().all(|x| x.is_finite()));

    let history_distance = l2_distance(&a, &b);
    assert!(
        history_distance > 1e-5,
        "identical probe lost all measurable history dependence: distance={history_distance}"
    );
}

#[test]
fn recurrent_reset_selectively_erases_history_dependence() {
    let dim = CfCNetworkConfig::default().input_dim;

    let retained_a = state_after_history(&history_a(dim), false);
    let retained_b = state_after_history(&history_b(dim), false);
    let reset_a = state_after_history(&history_a(dim), true);
    let reset_b = state_after_history(&history_b(dim), true);

    let retained_distance = l2_distance(&retained_a, &retained_b);
    let reset_distance = l2_distance(&reset_a, &reset_b);

    assert!(retained_distance > 1e-5);
    assert!(
        reset_distance <= retained_distance * 0.01 + 1e-8,
        "reset did not selectively collapse history signal: retained={retained_distance}, reset={reset_distance}"
    );
}

#[test]
fn reset_control_holds_weights_and_current_probe_constant() {
    let dim = CfCNetworkConfig::default().input_dim;
    let reset_a = state_after_history(&history_a(dim), true);
    let reset_b = state_after_history(&history_b(dim), true);

    // Both networks use the exact same genesis-derived weights and receive the
    // same current probe after state reset. Any residual difference would mean
    // the purported state-reset control is not actually isolating recurrence.
    assert!(
        l2_distance(&reset_a, &reset_b) < 1e-7,
        "deterministic reset control retained unexplained history information"
    );
}
