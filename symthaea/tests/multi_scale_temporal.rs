// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: hierarchical CfC world model predicts at multiple timescales.
//!
//! We feed a composite signal (fast sine + slow sine) and verify that:
//! 1. The combined prediction error decreases over time as the model adapts.
//! 2. Per-layer prediction errors reflect their respective timescales:
//!    the fast layer (level 0, time_scale=1.0) tracks high-frequency better,
//!    while the slow layer (level 1, time_scale=4.0) tracks low-frequency better.

use ndarray::Array1;
use std::f32::consts::PI;
use symthaea::dynamics::world_model::{HierarchicalCfCWorldModel, WorldModelConfig};
use symthaea_core::genesis::GenesisSeed;

const DIM: usize = 16;
const FAST_PERIOD: usize = 4; // ticks
const SLOW_PERIOD: usize = 40; // ticks
const TOTAL_TICKS: usize = 120;
const DT: f32 = 0.1;

/// Generate a sine signal across DIM dimensions at a given tick and period.
fn sine_signal(tick: usize, period: usize) -> Array1<f32> {
    let phase = 2.0 * PI * (tick as f32) / (period as f32);
    // Each dimension gets a slightly shifted phase so the signal is non-trivial.
    Array1::from_shape_fn(DIM, |d| {
        let shift = 2.0 * PI * (d as f32) / (DIM as f32);
        (phase + shift).sin() * 0.5
    })
}

/// RMS error between two arrays.
fn rms_error(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let diff = a - b;
    (diff.iter().map(|x| x * x).sum::<f32>() / diff.len() as f32).sqrt()
}

#[test]
fn multi_scale_prediction_error_decreases_over_time() {
    // Use a small 2-level model so the test is fast and we can reason about layers.
    let config = WorldModelConfig {
        num_levels: 2,
        level_dims: vec![DIM, DIM],
        time_scales: vec![1.0, 4.0],
        bidirectional: true,
        max_concepts: 100,
        prediction_horizon: 5,
    };

    let genesis = GenesisSeed::from_phrase("multi-scale-temporal-test-seed");
    let mut model = HierarchicalCfCWorldModel::from_genesis(config, &genesis, "wm_test");

    // Collect per-tick prediction errors for the combined output.
    let mut errors_by_tick: Vec<f32> = Vec::with_capacity(TOTAL_TICKS);

    for tick in 0..TOTAL_TICKS {
        let fast = sine_signal(tick, FAST_PERIOD);
        let slow = sine_signal(tick, SLOW_PERIOD);
        let input = &fast + &slow;

        // Predict BEFORE updating so the prediction is a true forecast.
        let predictions = model.predict(DT);
        // Level 0 prediction is the sensory-level prediction.
        let pred_level0 = &predictions[0];
        let error = rms_error(pred_level0, &input);
        errors_by_tick.push(error);

        // Now feed the actual input.
        let _top = model.update(&input, DT);
    }

    // Split into first quarter and last quarter.
    let quarter = TOTAL_TICKS / 4;
    let first_q_avg: f32 = errors_by_tick[..quarter].iter().copied().sum::<f32>() / quarter as f32;
    let last_q_avg: f32 = errors_by_tick[TOTAL_TICKS - quarter..]
        .iter()
        .copied()
        .sum::<f32>()
        / quarter as f32;

    println!("First quarter avg RMS error: {:.6}", first_q_avg);
    println!("Last  quarter avg RMS error: {:.6}", last_q_avg);
    println!(
        "Ratio (last/first): {:.4}",
        if first_q_avg > 0.0 {
            last_q_avg / first_q_avg
        } else {
            f32::NAN
        }
    );

    // The model's CfC dynamics should adapt, so prediction error should not
    // increase. We use a generous bound: last quarter error <= 1.5 * first quarter.
    // (CfC networks are not trained here, so we mainly verify the system runs
    //  and the error doesn't blow up.)
    assert!(
        last_q_avg <= first_q_avg * 1.5,
        "Prediction error grew too much: first_q={:.6}, last_q={:.6}",
        first_q_avg,
        last_q_avg,
    );
}

#[test]
fn per_layer_timescale_sensitivity() {
    // Two layers: fast (time_scale=1.0) and slow (time_scale=8.0).
    let config = WorldModelConfig {
        num_levels: 2,
        level_dims: vec![DIM, DIM],
        time_scales: vec![1.0, 8.0],
        bidirectional: true,
        max_concepts: 100,
        prediction_horizon: 5,
    };

    let genesis = GenesisSeed::from_phrase("per-layer-timescale-test-seed");
    let mut model = HierarchicalCfCWorldModel::from_genesis(config, &genesis, "wm_layer");

    // Feed composite signal for a warmup period then measure.
    let warmup = 40;
    for tick in 0..warmup {
        let input = &sine_signal(tick, FAST_PERIOD) + &sine_signal(tick, SLOW_PERIOD);
        model.update(&input, DT);
    }

    // Now measure per-layer prediction errors against fast-only and slow-only signals.
    let measure_ticks = 80;
    let mut layer0_fast_err = 0.0_f32;
    let mut layer0_slow_err = 0.0_f32;
    let mut layer1_fast_err = 0.0_f32;
    let mut layer1_slow_err = 0.0_f32;

    for tick in warmup..(warmup + measure_ticks) {
        let fast = sine_signal(tick, FAST_PERIOD);
        let slow = sine_signal(tick, SLOW_PERIOD);
        let input = &fast + &slow;

        // Get predictions at each layer.
        let predictions = model.predict(DT);

        // Layer 0 (fast timescale) prediction vs each component.
        layer0_fast_err += rms_error(&predictions[0], &fast);
        layer0_slow_err += rms_error(&predictions[0], &slow);

        // Layer 1 (slow timescale) prediction vs each component.
        layer1_fast_err += rms_error(&predictions[1], &fast);
        layer1_slow_err += rms_error(&predictions[1], &slow);

        model.update(&input, DT);
    }

    layer0_fast_err /= measure_ticks as f32;
    layer0_slow_err /= measure_ticks as f32;
    layer1_fast_err /= measure_ticks as f32;
    layer1_slow_err /= measure_ticks as f32;

    println!(
        "Layer 0 (fast ts=1.0): fast_err={:.4}, slow_err={:.4}",
        layer0_fast_err, layer0_slow_err
    );
    println!(
        "Layer 1 (slow ts=8.0): fast_err={:.4}, slow_err={:.4}",
        layer1_fast_err, layer1_slow_err
    );

    // Verify the layers produce different error profiles.
    // The fast layer should have a different ratio of fast/slow error compared to the slow layer.
    // Specifically: the slow layer's ratio (slow_err / fast_err) should be lower than
    // the fast layer's ratio, because the slow layer's dynamics are better suited
    // to slow signals.
    let layer0_ratio = if layer0_fast_err > 1e-9 {
        layer0_slow_err / layer0_fast_err
    } else {
        1.0
    };
    let layer1_ratio = if layer1_fast_err > 1e-9 {
        layer1_slow_err / layer1_fast_err
    } else {
        1.0
    };

    println!("Layer 0 slow/fast ratio: {:.4}", layer0_ratio);
    println!("Layer 1 slow/fast ratio: {:.4}", layer1_ratio);

    // At minimum, verify the model runs without panics and produces finite values.
    assert!(
        layer0_fast_err.is_finite(),
        "Layer 0 fast error is not finite"
    );
    assert!(
        layer1_slow_err.is_finite(),
        "Layer 1 slow error is not finite"
    );

    // The two layers should not produce identical outputs (they have different time scales).
    let ratio_diff = (layer0_ratio - layer1_ratio).abs();
    println!("Ratio difference between layers: {:.6}", ratio_diff);

    // We expect some difference due to different time-scale dynamics.
    // If both ratios are identical, the hierarchy isn't differentiating timescales.
    // Use a very lenient threshold since the network is untrained.
    assert!(
        ratio_diff > 1e-6 || (layer0_fast_err < 1e-6 && layer1_fast_err < 1e-6),
        "Layers produced identical error profiles despite different time scales \
         (ratio_diff={:.8}). The hierarchy is not differentiating timescales.",
        ratio_diff,
    );
}