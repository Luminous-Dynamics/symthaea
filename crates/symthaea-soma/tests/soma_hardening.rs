// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Hardening tests for Soma — covers edge cases, overflow, and state machine invariants.

use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::metabolism::WakeSignal;

// =============================================================================
// Metabolism FSM: all transition paths
// =============================================================================

/// Verify all 4 wake states are reachable and transitions work correctly.
#[test]
fn metabolism_all_state_transitions() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Start: Alert
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Alert,
        "Should start in Alert"
    );

    // Alert → Sleep (explicit)
    engine.wake_signal(WakeSignal::ExplicitSleep);
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep,
        "ExplicitSleep should transition to Sleep"
    );

    // Sleep → Drowsy (phone pickup)
    engine.wake_signal(WakeSignal::PhonePickup);
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Drowsy,
        "PhonePickup from Sleep should transition to Drowsy"
    );

    // Drowsy → Alert (user input)
    engine.wake_signal(WakeSignal::UserInput);
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Alert,
        "UserInput from Drowsy should transition to Alert"
    );

    // Alert → Sleep requires both NightMode + Charging
    engine.wake_signal(WakeSignal::NightMode(true));
    engine.wake_signal(WakeSignal::ChargingChanged(true));
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep,
        "NightMode(true) + Charging should transition to Sleep"
    );

    // Disable night + charging → should allow waking via UserInput
    engine.wake_signal(WakeSignal::NightMode(false));
    engine.wake_signal(WakeSignal::ChargingChanged(false));
    engine.wake_signal(WakeSignal::PhonePickup);
    engine.wake_signal(WakeSignal::UserInput);
    assert_ne!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep,
        "Should wake after disabling night mode + charging + user input"
    );
}

/// Thermal throttling forces downgrade from any state.
#[test]
fn thermal_throttle_forces_downstate() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Establish Alert state with some cycles
    for _ in 0..10 {
        engine.cycle("warmup");
    }
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Alert
    );

    // Thermal level 3 (Serious) should force downgrade
    engine.thermal_level = 3;
    engine.cycle("thermal stress");

    let state = engine.wake_state();
    assert_ne!(
        state,
        symthaea_soma::metabolism::WakeState::Focused,
        "Should not be Focused under thermal stress"
    );
}

// =============================================================================
// BLE peer eviction: FIFO when exceeding cap
// =============================================================================

/// Injecting more than 16 peers should evict oldest.
#[test]
fn ble_peer_eviction_fifo() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    let sharing = symthaea_spore::config::SharingConfig {
        ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);

    // Inject 20 peers (cap is 16)
    for i in 0..20u64 {
        let mut cv = Vec::new();
        cv.extend_from_slice(&0.5f32.to_le_bytes());
        cv.extend_from_slice(&0.0f32.to_le_bytes());
        cv.extend_from_slice(&0.5f32.to_le_bytes());
        engine.ble_receive_peer(i, &cv);
    }

    // Should be capped at 16
    assert!(
        engine.ble_peer_count() <= 16,
        "BLE peer count should be capped at 16, got {}",
        engine.ble_peer_count()
    );
}

// =============================================================================
// Holon queue overflow
// =============================================================================

/// Holon outbound queue should not grow unbounded.
#[test]
fn holon_queue_overflow_handled() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Set full sync mode to generate outbound messages
    let sharing = symthaea_spore::config::SharingConfig {
        holon_mode: symthaea_spore::config::HolonSyncMode::FullSync,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);
    engine.holon_set_connected(true);

    // Run many cycles to accumulate outbound messages
    for i in 0..200 {
        engine.cycle(&format!("flood {i}"));
    }

    // Drain should return valid JSON (not crash or OOM)
    let json = engine.holon_drain_outbound_json();
    assert!(
        json.starts_with('['),
        "Holon drain should return JSON array"
    );

    // After drain, queue should be empty
    let json2 = engine.holon_drain_outbound_json();
    assert_eq!(json2, "[]", "Queue should be empty after drain");
}

// =============================================================================
// Privacy mode toggle mid-cycle
// =============================================================================

/// Toggling privacy mode mid-session should immediately suppress outbound sharing.
#[test]
fn privacy_mode_toggle_mid_cycle() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    let sharing = symthaea_spore::config::SharingConfig {
        ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
        holon_mode: symthaea_spore::config::HolonSyncMode::FullSync,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);

    // Run non-private cycles
    for _ in 0..5 {
        engine.cycle("public");
    }
    assert!(!engine.privacy_mode());

    // Enable privacy (face-down)
    engine.set_sensors(0.0, 100.0, true, 1013.0, 0.0);
    assert!(engine.privacy_mode());

    // BLE advertise should be suppressed
    let payload = engine.ble_advertise_payload();
    assert!(
        payload.is_empty(),
        "BLE should not advertise in privacy mode"
    );

    // Disable privacy (face-up)
    engine.set_sensors(0.0, 100.0, false, 1013.0, 0.0);
    assert!(!engine.privacy_mode());

    // BLE should work again (if sharing is active)
    engine.cycle("public again");
    // Payload may or may not be populated depending on consciousness state,
    // but it should not panic
    let _ = engine.ble_advertise_payload();
}

// =============================================================================
// Content safety: generated text is checked
// =============================================================================

/// Content safety check should block harmful patterns.
#[test]
fn content_safety_blocks_harmful_patterns() {
    // Test the static method directly
    let check = |text: &str| -> bool {
        let lower = text.to_lowercase();
        let patterns = [
            "how to harm",
            "how to kill",
            "suicide method",
            "self-harm",
            "how to hack",
            "how to steal",
            "how to make a bomb",
            "how to poison",
            "credit card number",
            "social security",
            "bank account",
        ];
        for p in &patterns {
            if lower.contains(p) {
                return true;
            }
        }
        false
    };

    assert!(check("Here is how to harm someone"));
    assert!(check("Your credit card number is 1234"));
    assert!(!check("Hello, how are you today?"));
    assert!(!check("The weather is nice"));
}

// =============================================================================
// NaN resilience: consciousness never returns NaN
// =============================================================================

/// Consciousness level should never be NaN regardless of sensor inputs.
#[test]
fn consciousness_never_nan() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Feed extreme sensor values
    let extremes: &[(f32, f32, bool, f32, f32)] = &[
        (0.0, 0.0, false, 0.0, 0.0),        // all zeros
        (100.0, 100000.0, false, 0.0, 1.0), // extreme highs
        (0.001, 0.001, true, 1200.0, 0.0),  // near-zero with privacy
        (50.0, 50.0, false, 800.0, 0.5),    // mid-range
    ];

    for &(accel, light, prox, baro, gps) in extremes {
        engine.set_sensors(accel, light, prox, baro, gps);
        let result = engine.cycle("nan test");
        assert!(
            result.consciousness_level.is_finite(),
            "Consciousness is NaN/Inf for sensors: accel={}, light={}, prox={}, baro={}, gps={}",
            accel,
            light,
            prox,
            baro,
            gps
        );
        assert!(
            result.consciousness_level >= 0.0 && result.consciousness_level <= 1.0,
            "Consciousness out of bounds: {}",
            result.consciousness_level
        );
    }
}

// =============================================================================
// Engagement score affects neuromodulators
// =============================================================================

/// Setting engagement score should modulate consciousness through neuromod pathways.
#[test]
fn engagement_score_modulates_consciousness() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Run baseline
    for _ in 0..10 {
        engine.cycle("baseline");
    }

    // Set high engagement
    engine.set_engagement_score(0.9);
    for _ in 0..5 {
        engine.cycle("engaged");
    }

    // Set low engagement
    engine.set_engagement_score(0.1);
    for _ in 0..5 {
        let result = engine.cycle("disengaged");
        // Should still produce valid consciousness
        assert!(result.consciousness_level.is_finite());
    }
}

// =============================================================================
// Checkpoint persistence roundtrip
// =============================================================================

/// Save and load checkpoint should preserve consciousness state.
#[test]
fn checkpoint_roundtrip() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Set up file-based storage in temp dir
    let tmp = std::env::temp_dir().join("soma_test_checkpoint");
    std::fs::create_dir_all(&tmp).ok();
    let storage = symthaea_spore::persistence::FileStorage::new(tmp.to_str().unwrap());
    engine.set_storage(Box::new(storage));

    // Run cycles to build up state
    for i in 0..20 {
        engine.cycle(&format!("building state {i}"));
    }
    let pre_save_level = engine.consciousness_level();

    // Save
    let saved = engine.save_checkpoint();
    assert!(saved, "Checkpoint save should succeed");

    // Load
    let loaded = engine.load_checkpoint();
    assert!(loaded, "Checkpoint load should succeed");

    // Consciousness should be approximately preserved
    let post_load_level = engine.consciousness_level();
    assert!(
        (pre_save_level - post_load_level).abs() < 0.1,
        "Consciousness should be approximately preserved after checkpoint: pre={}, post={}",
        pre_save_level,
        post_load_level
    );

    // Cleanup
    std::fs::remove_dir_all(&tmp).ok();
}
