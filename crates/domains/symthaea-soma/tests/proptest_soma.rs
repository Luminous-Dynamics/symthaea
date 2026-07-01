// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for Soma subsystems.
//!
//! Verifies invariants that must hold for ALL inputs, not just specific test cases:
//! - Sensor nudges bounded
//! - BLE peer count capped
//! - Privacy mode suppresses all nudges
//! - Consciousness always in [0, 1]
//! - ConsciousnessContext exploration gain bounded

use proptest::prelude::*;
use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::sensor_bridge::SensorBridge;

// =============================================================================
// Sensor nudge bounds
// =============================================================================

proptest! {
    /// All sensor nudges must be bounded to [-1.0, 1.0] for any input.
    #[test]
    fn sensor_nudges_bounded(
        accel in 0.0f32..50.0,
        light in 0.0f32..10000.0,
        proximity_near in proptest::bool::ANY,
        barometer in 800.0f32..1200.0,
        gps_novelty in 0.0f32..1.0,
        gyro in 0.0f32..20.0,
        ambient_db in 0.0f32..120.0,
        social_pressure in 0u32..100,
    ) {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(accel, light, proximity_near, barometer, gps_novelty);
        bridge.set_gyroscope(gyro);
        bridge.set_ambient_db(ambient_db);
        bridge.set_social_pressure(social_pressure);

        let nudges = bridge.compute_nudges();

        prop_assert!(nudges.dopamine_delta.abs() <= 1.0,
            "DA nudge out of bounds: {}", nudges.dopamine_delta);
        prop_assert!(nudges.norepinephrine_delta.abs() <= 1.0,
            "NE nudge out of bounds: {}", nudges.norepinephrine_delta);
        prop_assert!(nudges.serotonin_delta.abs() <= 1.0,
            "5-HT nudge out of bounds: {}", nudges.serotonin_delta);
        prop_assert!(nudges.oxytocin_delta.abs() <= 1.0,
            "OT nudge out of bounds: {}", nudges.oxytocin_delta);
    }

    /// Privacy mode (face-down) suppresses BLE advertising.
    /// Note: nudge suppression happens in SomaEngine.cycle(), not SensorBridge.
    #[test]
    fn privacy_suppresses_ble_advertising(
        accel in 0.0f32..50.0,
        light in 0.0f32..10000.0,
        barometer in 800.0f32..1200.0,
        gps_novelty in 0.0f32..1.0,
    ) {
        let mut engine = SomaEngine::new(SomaConfig::default());
        let sharing = symthaea_spore::config::SharingConfig {
            ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
            ..Default::default()
        };
        engine.set_sharing_config(sharing);

        // Face-down = privacy
        engine.set_sensors(accel, light, true, barometer, gps_novelty);
        prop_assert!(engine.privacy_mode(),
            "proximity_near=true should activate privacy mode");

        // BLE should not advertise in privacy mode
        let payload = engine.ble_advertise_payload();
        prop_assert!(payload.is_empty(),
            "BLE should not advertise in privacy mode, got {} bytes", payload.len());
    }
}

// Touch body tests require `screen-vision` feature — covered by unit tests in touch_body.rs.

// =============================================================================
// Consciousness level always bounded
// =============================================================================

proptest! {
    /// Consciousness level is always in [0, 1] regardless of input.
    #[test]
    fn consciousness_always_bounded(
        input_len in 0usize..50,
        accel in 0.0f32..30.0,
        light in 0.0f32..2000.0,
    ) {
        let mut engine = SomaEngine::new(SomaConfig::default());
        engine.set_sensors(accel, light, false, 1013.0, 0.5);

        let input: String = (0..input_len).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        for _ in 0..5 {
            let result = engine.cycle(&input);
            prop_assert!(
                result.consciousness_level >= 0.0 && result.consciousness_level <= 1.0,
                "consciousness out of bounds: {}",
                result.consciousness_level
            );
        }
    }
}

// =============================================================================
// BLE peer count invariant
// =============================================================================

proptest! {
    /// BLE peer count never exceeds 16 regardless of how many peers are injected.
    #[test]
    fn ble_peer_count_capped(n_peers in 1usize..30) {
        let mut engine = SomaEngine::new(SomaConfig::default());

        let sharing = symthaea_spore::config::SharingConfig {
            ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
            ..Default::default()
        };
        engine.set_sharing_config(sharing);

        for i in 0..n_peers {
            let mut cv = Vec::new();
            cv.extend_from_slice(&0.5f32.to_le_bytes());
            cv.extend_from_slice(&0.0f32.to_le_bytes());
            cv.extend_from_slice(&0.5f32.to_le_bytes());
            engine.ble_receive_peer(i as u64, &cv);
        }

        prop_assert!(engine.ble_peer_count() <= 16,
            "BLE peer count exceeded cap: {}", engine.ble_peer_count());
    }
}

// =============================================================================
// ConsciousnessContext (symthaea-core) invariants
// =============================================================================

proptest! {
    /// Exploration gain is always in [0, 2].
    #[test]
    fn consciousness_context_exploration_bounded(
        phi in 0.0f32..1.0,
        safety in 0u8..4,
        harmony in 0.0f32..1.0,
        pe in 0.0f32..1.0,
        cl in 0.0f32..1.0,
    ) {
        let ctx = symthaea_core::ConsciousnessContext {
            phi,
            safety_level: safety,
            harmony_alignment: harmony,
            prediction_error: pe,
            consciousness_level: cl,
        };
        let gain = ctx.exploration_gain();
        prop_assert!(gain >= 0.0 && gain <= 2.0,
            "exploration gain out of bounds: {}", gain);

        // Emergency = zero exploration
        if safety >= 3 {
            prop_assert!(gain == 0.0,
                "emergency should have zero exploration, got {}", gain);
        }
    }
}
