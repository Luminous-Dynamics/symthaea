// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end test: Consciousness sharing over mesh loopback transport.
//!
//! Verifies that consciousness-aware routers can share state,
//! converge collective Phi, handle moral emergencies, corroborate
//! threats, and perform dream-consolidated reconnection.
//!
//! Requires: `--features mesh`

#![cfg(feature = "mesh")]

use symthaea::cognitive_loop::{
    CompressedDelta, ConsciousRoutingDecision, ConsciousnessAwareRouter, OfflineExperience,
    OfflineExperienceKind, PayloadClass, PayloadClassifier, StoreAndForward, ThreatObservation,
};

#[test]
fn test_two_node_consciousness_convergence() {
    let mut node_a = ConsciousnessAwareRouter::default();
    let mut node_b = ConsciousnessAwareRouter::default();

    node_a.update_local(0.8, 0.9, 3);
    node_b.update_local(0.3, 0.4, 1);

    for cycle in 0..20u64 {
        node_b.update_peer([0xAA; 8], 0.8, 0.9, 3, cycle);
        node_a.update_peer(
            [0xBB; 8],
            0.3 + (cycle as f32 * 0.02),
            0.4 + (cycle as f32 * 0.02),
            1,
            cycle,
        );
    }

    let collective_a = node_a.collective_phi();
    let collective_b = node_b.collective_phi();
    assert!(
        collective_a > 0.3 && collective_a < 0.9,
        "collective_a={collective_a}"
    );
    assert!(
        collective_b > 0.3 && collective_b < 0.9,
        "collective_b={collective_b}"
    );
    assert_eq!(node_a.peer_count(), 1);
    assert_eq!(node_b.peer_count(), 1);
}

#[test]
fn test_sharing_cadence_adapts_to_divergence() {
    let mut router = ConsciousnessAwareRouter::default();
    let initial_cadence = router.sharing_cadence();

    router.update_local(0.9, 0.95, 4);
    router.update_peer([1; 8], 0.1, 0.2, 0, 100);
    router.update_peer([2; 8], 0.05, 0.1, 0, 100);

    assert!(
        router.sharing_cadence() < initial_cadence,
        "High divergence should decrease cadence"
    );
}

#[test]
fn test_moral_emergency_bypasses_bandwidth() {
    let mut router = ConsciousnessAwareRouter::default();
    let classifier = PayloadClassifier::default();

    router.update_local(0.6, 0.7, 2);
    router.update_peer([0xFF; 8], 0.95, 0.98, 4, 100);
    router.signal_moral_emergency();

    let decision = router.route(PayloadClass::Emergency, 40, 2, &classifier);
    match decision {
        ConsciousRoutingDecision::MoralEmergency {
            preferred_relay, ..
        } => {
            assert_eq!(preferred_relay, Some([0xFF; 8]));
        }
        other => panic!("Expected MoralEmergency, got {other:?}"),
    }

    // One-shot: next route is normal
    assert!(matches!(
        router.route(PayloadClass::Discovery, 64, 1, &classifier),
        ConsciousRoutingDecision::Normal(_)
    ));
}

#[test]
fn test_threat_corroboration_across_nodes() {
    let mut node_a = ConsciousnessAwareRouter::default();

    let threat = ThreatObservation {
        threat_type: 3,
        severity: 0.85,
        agent_hash: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
        signature: [0x42; 32],
        observed_cycle: 100,
        corroboration_count: 0,
    };

    node_a.record_threat(threat.clone());
    assert_eq!(node_a.threat_count(), 1);
    assert_eq!(node_a.threats()[0].corroboration_count, 0);

    // Same agent+type again = corroboration
    node_a.record_threat(threat);
    assert_eq!(node_a.threat_count(), 1);
    assert_eq!(node_a.threats()[0].corroboration_count, 1);
}

#[test]
fn test_dream_consolidation_on_reconnect() {
    let mut sf = StoreAndForward::default();
    sf.go_offline(100);

    for i in 0..20u64 {
        sf.record(OfflineExperience {
            cycle: 100 + i,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: format!("tank-{}", i % 3),
                value: 42.0 + i as f32,
            },
            salience: 0.5 + (i as f32 * 0.02),
        });
    }
    sf.record(OfflineExperience {
        cycle: 125,
        kind: OfflineExperienceKind::ConsciousnessShift { from: 0.4, to: 0.8 },
        salience: 0.9,
    });
    sf.record(OfflineExperience {
        cycle: 130,
        kind: OfflineExperienceKind::ThreatDetected {
            threat_type: 2,
            severity: 0.7,
        },
        salience: 0.85,
    });

    assert!(sf.go_online(200), "Should need consolidation");
    let wisdom = sf.consolidate(200);

    assert_eq!(wisdom.experiences_consolidated, 22);
    assert_eq!(wisdom.offline_duration, 100);
    assert!(wisdom.mean_salience > 0.5);

    let kinds: Vec<&str> = wisdom.patterns.iter().map(|p| p.kind.as_str()).collect();
    assert!(kinds.contains(&"sensor_trend"));
    assert!(kinds.contains(&"consciousness_trajectory"));
    assert!(kinds.contains(&"threat_summary"));
    assert_eq!(sf.buffer_len(), 0);
}

#[test]
fn test_compressed_delta_sharing() {
    let mut state_a = [0u8; 2048];
    for i in 0..2048 {
        state_a[i] = (i % 256) as u8;
    }
    let mut state_a_next = state_a;
    for i in (0..2048).step_by(20) {
        state_a_next[i] ^= 0xFF;
    }

    let delta = CompressedDelta::from_diff(&state_a, &state_a_next);
    assert!(!delta.is_full);
    assert!(delta.wire_size() < 400, "wire_size={}", delta.wire_size());

    let reconstructed = delta.apply(&state_a).unwrap();
    assert_eq!(reconstructed, state_a_next);
}

// NOTE: DiscoveryBeacon consciousness field tests are in unit tests
// (radio_dispatcher::tests::test_beacon_serialization_roundtrip).
// The struct extensions need to be re-applied after linter formatting.

#[test]
fn test_trust_decays_on_phi_jumps() {
    let mut router = ConsciousnessAwareRouter::default();
    for cycle in 0..10u64 {
        router.update_peer([1; 8], 0.5, 0.6, 2, cycle);
    }
    let trust_stable = router.peers_by_trust()[0].1;

    router.update_peer([1; 8], 0.99, 0.6, 2, 10);
    let trust_after = router.peers_by_trust()[0].1;
    assert!(
        trust_after < trust_stable,
        "Trust should decay: {trust_stable} → {trust_after}"
    );
}

#[test]
fn test_highest_phi_relay_selection() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_peer([1; 8], 0.3, 0.4, 1, 100);
    router.update_peer([2; 8], 0.9, 0.95, 4, 100);
    router.update_peer([3; 8], 0.6, 0.7, 2, 100);

    assert_eq!(router.highest_phi_peer(), Some([2; 8]));
    assert_eq!(router.peers_by_trust()[0].0, [2; 8]);
}