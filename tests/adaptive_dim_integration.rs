// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: adaptive HDC dimensionality resize lifecycle.
//!
//! Verifies the full loop: start small -> scale up under high error ->
//! scale back down when error drops -> bridge still functional after resizing.

use ndarray::Array1;
use symthaea::hdc_ltc_bridge::{AdaptiveDimConfig, HdcLtcBridge, HdcLtcBridgeConfig};

fn make_bridge() -> HdcLtcBridge {
    let config = HdcLtcBridgeConfig {
        hdc_dim: 2048,
        adaptive_dim: Some(AdaptiveDimConfig {
            min_dim: 2048,
            max_dim: 8192,
            upscale_error_threshold: 0.7,
            downscale_error_threshold: 0.2,
            scale_step: 2048,
            cooldown_cycles: 5,
        }),
        ..HdcLtcBridgeConfig::default()
    };
    HdcLtcBridge::new(config)
}

#[test]
fn test_adaptive_dim_full_lifecycle() {
    let mut bridge = make_bridge();
    let min_dim = 2048;
    let max_dim = 8192;

    // Initial dimension is min_dim
    assert_eq!(bridge.current_hdc_dim(), min_dim, "should start at min_dim");

    // Phase 1: high error -> scale UP
    for _ in 0..30 {
        let dim = bridge.current_hdc_dim();
        assert!(
            dim >= min_dim && dim <= max_dim,
            "dim {} out of bounds",
            dim
        );
        bridge.maybe_resize(0.9);
    }
    let dim_after_up = bridge.current_hdc_dim();
    assert!(
        dim_after_up > min_dim,
        "expected dim to increase above {}, got {}",
        min_dim,
        dim_after_up
    );
    println!("after upscale phase: dim = {}", dim_after_up);

    // Phase 2: low error -> scale DOWN
    for _ in 0..30 {
        let dim = bridge.current_hdc_dim();
        assert!(
            dim >= min_dim && dim <= max_dim,
            "dim {} out of bounds",
            dim
        );
        bridge.maybe_resize(0.1);
    }
    let dim_after_down = bridge.current_hdc_dim();
    assert!(
        dim_after_down < dim_after_up,
        "expected dim to decrease from {}, got {}",
        dim_after_up,
        dim_after_down
    );
    println!("after downscale phase: dim = {}", dim_after_down);

    // Final bounds check
    assert!(
        dim_after_down >= min_dim && dim_after_down <= max_dim,
        "final dim {} out of bounds",
        dim_after_down
    );

    // Bridge still functional after resizing: run a forward pass
    let input = Array1::from_vec(vec![0.5; bridge.config().input_dim]);
    let output = bridge.forward(&input, 0.02);
    assert_eq!(
        output.len(),
        bridge.config().output_dim,
        "forward() output dimension mismatch after resize"
    );
    // Output should contain finite values
    assert!(
        output.iter().all(|v| v.is_finite()),
        "forward() produced non-finite values after resize"
    );
    println!(
        "forward() works after resize lifecycle: output len = {}",
        output.len()
    );
}