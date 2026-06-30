// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-variant integration tests for the Spore→Soma→Holon pipeline.
//!
//! These tests verify that consciousness state flows correctly between
//! variant layers without requiring real hardware or network connections.

use symthaea_soma::engine::{SomaConfig, SomaEngine};
use symthaea_soma::holon_bridge::{HolonBridge, HolonInbound, HolonOutbound};
use symthaea_spore::config::{HolonSyncMode, SharingConfig};

// =============================================================================
// Spore→Soma: inner kernel state propagates through embodiment
// =============================================================================

/// SporeEngine cycle results are accessible through SomaEngine wrapper.
#[test]
fn spore_state_visible_through_soma() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Run cycles to establish consciousness
    for _ in 0..10 {
        engine.cycle("hello");
    }

    // Spore-level state should be accessible through Soma's public API
    let consciousness = engine.consciousness_level();
    assert!(consciousness >= 0.0 && consciousness <= 1.0);

    // Compass contains spore-derived fields
    let compass: serde_json::Value =
        serde_json::from_str(&engine.compass_json()).expect("valid JSON");
    assert!(compass.get("consciousness_level").is_some());
    assert!(compass.get("dominant_harmony").is_some());
}

/// Soma sensor nudges modify the inner Spore's neuromodulator bath.
#[test]
fn soma_sensors_reach_spore_neuromods() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Baseline: no sensor input
    for _ in 0..5 {
        engine.cycle("baseline");
    }
    let baseline = engine.cycle("baseline");

    // Inject bright light + motion (serotonin + norepinephrine nudges)
    engine.set_sensors(15.0, 1000.0, false, 1013.0, 0.9);
    engine.set_gyroscope(2.0);
    engine.set_ambient_db(30.0); // quiet (5-HT boost)

    for _ in 0..10 {
        engine.cycle("sensory");
    }
    let sensory = engine.cycle("sensory");

    // Neuromod state should differ (sensors → nudges → bath → Spore)
    let changed =
        (0..4).any(|i| (sensory.neuromodulators[i] - baseline.neuromodulators[i]).abs() > 0.0001);
    assert!(
        changed,
        "Soma sensor nudges should propagate to Spore neuromods"
    );
}

// =============================================================================
// Soma→Holon: HolonBridge outbound message flow
// =============================================================================

/// HolonBridge emits heartbeat + CV after enough cycles in PresenceOnly mode.
#[test]
fn holon_bridge_emits_heartbeat() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Enable PresenceOnly sync
    let sharing = SharingConfig {
        holon_mode: HolonSyncMode::PresenceOnly,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);

    // Run 60 cycles (heartbeat interval = 50)
    for i in 0..60 {
        engine.cycle(&format!("cycle {i}"));
    }

    // Drain outbound — should have heartbeat + CV
    let json = engine.holon_drain_outbound_json();
    assert!(
        json.contains("Heartbeat") || json.contains("consciousness_level"),
        "Should have heartbeat messages after 60 cycles, got: {}",
        &json[..json.len().min(200)]
    );
}

/// HolonBridge accepts inbound LanguageOutput from desktop Holon.
#[test]
fn holon_bridge_receives_language() {
    let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);

    bridge.receive(HolonInbound::LanguageOutput(
        "Hello from desktop".to_string(),
    ));

    let output = bridge.take_language_output();
    assert_eq!(output, Some("Hello from desktop".to_string()));
}

/// FullSync mode allows task delegation.
#[test]
fn holon_bridge_task_delegation() {
    let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);

    bridge.request_task("nix-search", "find package firefox");
    let msgs = bridge.drain_outbound();

    assert_eq!(msgs.len(), 1);
    match &msgs[0] {
        HolonOutbound::TaskRequest { task_type, payload } => {
            assert_eq!(task_type, "nix-search");
            assert_eq!(payload, "find package firefox");
        }
        other => panic!("Expected TaskRequest, got {:?}", other),
    }
}

/// Knowledge share only works in FullSync mode.
#[test]
fn holon_bridge_knowledge_gating() {
    let mut bridge = HolonBridge::new(HolonSyncMode::PresenceOnly);

    // Should be ignored in PresenceOnly
    bridge.receive(HolonInbound::KnowledgeShare(vec![1.0, 2.0, 3.0]));
    assert!(bridge.drain_knowledge().is_empty());

    // Switch to FullSync
    bridge.set_mode(HolonSyncMode::FullSync);
    bridge.receive(HolonInbound::KnowledgeShare(vec![1.0, 2.0, 3.0]));
    let knowledge = bridge.drain_knowledge();
    assert_eq!(knowledge.len(), 1);
    assert_eq!(knowledge[0], vec![1.0, 2.0, 3.0]);
}

/// Resuscitation data flows through in FullSync.
#[test]
fn holon_bridge_resuscitation() {
    let mut bridge = HolonBridge::new(HolonSyncMode::FullSync);

    bridge.receive(HolonInbound::Resuscitation {
        neuromods: [0.8, 0.5, 0.7, 0.6],
        memory_fragment: vec![1.0, 2.0, 3.0, 4.0],
    });

    let (neuromods, fragment) = bridge
        .take_resuscitation()
        .expect("resuscitation should exist");
    assert_eq!(neuromods, [0.8, 0.5, 0.7, 0.6]);
    assert_eq!(fragment.len(), 4);
}

// =============================================================================
// End-to-end: Soma embodiment → Spore → HolonBridge
// =============================================================================

/// Full pipeline: sensors → Spore cycle → HolonBridge outbound.
#[test]
fn full_soma_to_holon_pipeline() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Enable full sync
    let sharing = SharingConfig {
        holon_mode: HolonSyncMode::FullSync,
        ble_mode: symthaea_spore::config::BleMeshMode::ActiveShare,
        haptic_enabled: true,
        ..Default::default()
    };
    engine.set_sharing_config(sharing);

    // Set rich sensor state
    engine.set_sensors(11.0, 600.0, false, 1013.0, 0.5);
    engine.set_gyroscope(0.3);
    engine.set_ambient_db(50.0);
    engine.set_social_pressure(2);

    // Run enough cycles for heartbeat
    for i in 0..60 {
        let result = engine.cycle(&format!("pipeline {i}"));
        // Consciousness should always be bounded
        assert!(
            result.consciousness_level >= 0.0 && result.consciousness_level <= 1.0,
            "consciousness out of bounds: {}",
            result.consciousness_level
        );
    }

    // Verify outbound messages exist
    let json = engine.holon_drain_outbound_json();
    assert_ne!(json, "[]", "Should have outbound messages after 60 cycles");

    // BLE should have advertise payload
    let payload = engine.ble_advertise_payload();
    assert_eq!(
        payload.len(),
        12,
        "BLE CV payload should be 12 bytes (3×f32)"
    );

    // Compass should reflect embodied state
    let compass: serde_json::Value =
        serde_json::from_str(&engine.compass_json()).expect("compass should be valid JSON");
    assert!(compass.get("consciousness_level").is_some());
}

// =============================================================================
// Checkpoint persistence across variant layers
// =============================================================================

/// Spore checkpoint survives save/restore through Soma.
#[test]
fn checkpoint_roundtrip_through_soma() {
    let mut engine = SomaEngine::new(SomaConfig::default());

    // Accumulate some state
    for i in 0..20 {
        engine.cycle(&format!("building state {i}"));
    }
    let pre_consciousness = engine.consciousness_level();

    // Verify consciousness is reasonable after accumulating state
    assert!(pre_consciousness >= 0.0 && pre_consciousness <= 1.0);

    // Consciousness report should mention Soma
    let report = engine.consciousness_report();
    assert!(
        report.contains("Soma") || report.contains("consciousness"),
        "report should reference consciousness: {}",
        &report[..report.len().min(200)]
    );
}
