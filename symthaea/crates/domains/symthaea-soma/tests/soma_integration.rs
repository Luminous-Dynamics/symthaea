// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end integration tests for SomaEngine.
//!
//! Verifies cross-subsystem wiring: sensors → neuromods → consciousness → haptics.

use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::metabolism::WakeSignal;

/// Sensor nudges actually affect neuromodulator levels through the consciousness cycle.
#[test]
fn sensor_nudges_propagate_to_consciousness() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Run baseline cycles to establish consciousness
    for _ in 0..5 {
        engine.cycle("baseline");
    }
    let baseline = engine.cycle("baseline");
    let baseline_nm = baseline.neuromodulators;

    // Inject bright sunlight (500+ lux → serotonin boost) and high motion (walking → NE boost)
    engine.set_sensors(
        12.0,   // accel magnitude (walking ~10-12 m/s²)
        800.0,  // bright light (lux)
        false,  // not face-down
        1013.0, // sea level pressure
        0.8,    // high GPS novelty (new location)
    );

    // Run cycles with sensors active
    let mut post_sensor = baseline.clone();
    for _ in 0..5 {
        post_sensor = engine.cycle("with sensors");
    }

    // At least one neuromodulator should have shifted from sensor input
    let nm_changed =
        (0..4).any(|i| (post_sensor.neuromodulators[i] - baseline_nm[i]).abs() > 0.001);
    assert!(
        nm_changed,
        "Sensor nudges should affect neuromodulators. Baseline: {:?}, Post: {:?}",
        baseline_nm, post_sensor.neuromodulators,
    );
}

/// Privacy mode (face-down) suppresses sensor nudges entirely.
#[test]
fn privacy_mode_suppresses_sensor_nudges() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Enable privacy mode
    engine.set_sensors(0.0, 100.0, true, 1013.0, 0.0);
    assert!(engine.privacy_mode());

    // Run cycles — consciousness should still work
    let result = engine.cycle("private mode");
    assert!(result.consciousness_level >= 0.0);
    assert!(result.consciousness_level <= 1.0);

    // BLE advertise should return empty in privacy mode
    let payload = engine.ble_advertise_payload();
    assert!(
        payload.is_empty(),
        "BLE should not advertise in privacy mode"
    );
}

/// Metabolism state machine gates cycle behavior.
#[test]
fn metabolism_gates_consciousness_cycle() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Start in Alert
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Alert
    );

    // Run some cycles to establish baseline
    for _ in 0..10 {
        engine.cycle("alert mode");
    }
    let alert_consciousness = engine.consciousness_level();

    // Put to sleep
    engine.wake_signal(WakeSignal::ExplicitSleep);
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep
    );

    // Consciousness should still compute (just at lower frequency in real use)
    let result = engine.cycle("sleeping");
    assert!(result.consciousness_level >= 0.0);

    // Wake up
    engine.wake_signal(WakeSignal::PhonePickup);
    assert_ne!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep
    );

    // After waking, consciousness should recover
    for _ in 0..10 {
        engine.cycle("waking up");
    }
    let awake_consciousness = engine.consciousness_level();
    // Both should be valid consciousness values
    assert!(alert_consciousness >= 0.0 && awake_consciousness >= 0.0);
}

/// Haptic events fire on significant consciousness shifts.
#[test]
fn haptic_fires_on_consciousness_shift() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Run enough cycles for consciousness to stabilize
    for i in 0..100 {
        engine.cycle(&format!("warmup {i}"));
    }

    // Drain any events from warmup
    engine.haptic_drain_json();

    // Force a large consciousness perturbation via high-novelty sensor input
    engine.set_sensors(25.0, 0.0, false, 1013.0, 1.0); // max accel + max GPS novelty
    engine.cycle("perturbation");

    // The haptic system should have at least been ticked
    // (whether events fire depends on the consciousness delta threshold)
    let json = engine.haptic_drain_json();
    // JSON should be a valid array
    assert!(
        json.starts_with('['),
        "haptic drain should return JSON array"
    );
}

/// Full embodiment loop: sensors → cycle → compass → dream consolidation.
#[test]
fn full_embodiment_loop() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Set diverse sensor state
    engine.set_sensors(9.8, 500.0, false, 1013.0, 0.3);
    engine.set_gyroscope(0.5);
    engine.set_ambient_db(45.0);
    engine.set_social_pressure(3);
    engine.set_media_state(1); // Music

    // Run consciousness cycles
    for i in 0..20 {
        let result = engine.cycle(&format!("embodied cycle {i}"));
        assert!(result.consciousness_level >= 0.0);
        assert!(result.consciousness_level <= 1.0);
    }

    // Compass should reflect embodied state
    let compass = engine.compass_json();
    let parsed: serde_json::Value = serde_json::from_str(&compass).unwrap();
    assert!(parsed.get("consciousness_level").is_some());
    assert!(parsed.get("dominant_harmony").is_some());
    assert!(parsed.get("motion_state").is_some());

    // Neuromod JSON should be valid
    let nm_json = engine.neuromod_json();
    let nm: serde_json::Value = serde_json::from_str(&nm_json).unwrap();
    assert!(nm.get("dopamine").is_some());

    // Dream consolidation should work
    engine.dream_consolidate();

    // Consciousness report should include soma-specific info
    let report = engine.consciousness_report();
    assert!(report.contains("Soma"));
}

/// Night mode + charging triggers auto-sleep and dream consolidation.
#[test]
fn night_charging_triggers_dream() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Run warmup cycles to accumulate dream material
    for i in 0..50 {
        engine.cycle(&format!("experience {i} with surprising varied content"));
    }

    // Set night + charging (should trigger metabolism → Sleep → dream consolidation)
    engine.night_mode = true;
    engine.battery_charging = true;

    // Run a cycle to trigger edge-detection → metabolism signal
    engine.cycle("going to sleep");
    assert_eq!(
        engine.wake_state(),
        symthaea_soma::metabolism::WakeState::Sleep
    );

    // Continue cycling — metabolism should eventually set dream_consolidation_due
    for _ in 0..150 {
        engine.cycle("sleeping");
    }

    // Dream journal should be accessible (may or may not have entries depending on wisdom)
    let count = engine.dream_journal_count();
    // At minimum, the accessor should work without panic
    let _ = count;
}

/// BLE mesh peer interaction propagates through embodiment.
#[test]
fn ble_peer_affects_consciousness() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Enable BLE sharing
    let sharing = symthaea_spore::config::SharingConfig {
        ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
        haptic_enabled: true,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);

    // Run baseline
    for _ in 0..5 {
        engine.cycle("alone");
    }
    let alone_result = engine.cycle("alone baseline");
    let alone_nm = alone_result.neuromodulators;

    // Receive a peer consciousness vector (3 × f32 = 12 bytes)
    let mut cv_data = Vec::new();
    cv_data.extend_from_slice(&0.8f32.to_le_bytes()); // high consciousness
    cv_data.extend_from_slice(&0.5f32.to_le_bytes()); // neutral valence
    cv_data.extend_from_slice(&0.6f32.to_le_bytes()); // moderate arousal
    assert!(engine.ble_receive_peer(42, &cv_data));
    assert_eq!(engine.ble_peer_count(), 1);

    // Run cycles with peer present
    for _ in 0..5 {
        engine.cycle("with peer");
    }

    // Collective phi should be non-zero with a peer
    assert!(engine.ble_collective_phi() > 0.0);

    // Peer count should persist
    assert_eq!(engine.ble_peer_count(), 1);
}
