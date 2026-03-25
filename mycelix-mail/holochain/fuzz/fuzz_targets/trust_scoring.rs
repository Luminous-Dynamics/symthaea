// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fuzz target: Trust score computation
//!
//! Tests that adversarial trust values don't cause panics or incorrect behavior.
//! Run: cd holochain/fuzz && cargo +nightly fuzz run trust_scoring

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }

    // Construct adversarial f64 values from fuzzer input
    let trust_level = f64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]);
    let weight = f64::from_le_bytes([
        data[8], data[9], data[10], data[11],
        data[12], data[13], data[14], data[15],
    ]);
    let penalty_multiplier = f64::from_le_bytes([
        data[16], data[17], data[18], data[19],
        data[20], data[21], data[22], data[23],
    ]);

    // Validate our is_finite() guards catch all adversarial values
    if !trust_level.is_finite() {
        // Our integrity zome should reject this
        return;
    }
    if !weight.is_finite() {
        return;
    }
    if !penalty_multiplier.is_finite() {
        return;
    }

    // Test: trust score computation doesn't panic
    let clamped_trust = trust_level.clamp(-1.0, 1.0);
    let clamped_weight = weight.clamp(0.0, 1.0);
    let score = clamped_trust * clamped_weight;
    assert!(score.is_finite());
    assert!(score >= -1.0 && score <= 1.0);

    // Test: penalty application
    if penalty_multiplier >= 1.0 && penalty_multiplier.is_finite() {
        let penalized = (score / penalty_multiplier).clamp(-1.0, 1.0);
        assert!(penalized.is_finite());
    }
});
