//! End-to-end test: Consciousness sharing over mesh loopback transport.
//!
//! Verifies that two consciousness-aware routers can:
//! 1. Share consciousness state via compressed deltas
//! 2. Converge collective Phi toward a shared value
//! 3. Detect and respond to moral emergencies
//! 4. Perform dream-consolidated reconnection after offline periods
//! 5. Corroborate threat signatures across the mesh
//!
//! Requires: `--features mesh`

#![cfg(feature = "mesh")]

use symthaea::cognitive_loop::managers::radio_dispatcher::{
    CompressedDelta, ConsciousRoutingDecision, ConsciousnessAwareRouter, DiscoveryBeacon,
    OfflineExperience, OfflineExperienceKind, PayloadClass, PayloadClassifier, StoreAndForward,
    ThreatObservation,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Two-node consciousness convergence
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_two_node_consciousness_convergence() {
    let mut node_a = ConsciousnessAwareRouter::default();
    let mut node_b = ConsciousnessAwareRouter::default();

    // Node A has high consciousness, Node B starts low
    node_a.update_local(0.8, 0.9, 3); // Steward tier
    node_b.update_local(0.3, 0.4, 1); // Participant tier

    // Simulate 20 cycles of consciousness sharing
    for cycle in 0..20 {
        // Node A shares its state with Node B
        node_b.update_peer([0xAA; 8], 0.8, 0.9, 3, cycle);

        // Node B shares its state with Node A
        node_a.update_peer(
            [0xBB; 8],
            0.3 + (cycle as f32 * 0.02), // B's consciousness slowly rises
            0.4 + (cycle as f32 * 0.02),
            1,
            cycle,
        );
    }

    // Collective Phi should be between the two nodes' values
    let collective_a = node_a.collective_phi();
    let collective_b = node_b.collective_phi();

    assert!(
        collective_a > 0.3 && collective_a < 0.9,
        "Node A collective Phi {collective_a} should be between individual values"
    );
    assert!(
        collective_b > 0.3 && collective_b < 0.9,
        "Node B collective Phi {collective_b} should be between individual values"
    );

    // Both nodes should see each other as peers
    assert_eq!(node_a.peer_count(), 1);
    assert_eq!(node_b.peer_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Adaptive sharing cadence
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sharing_cadence_adapts_to_divergence() {
    let mut router = ConsciousnessAwareRouter::default();
    let initial_cadence = router.sharing_cadence();

    // Add highly divergent peers → cadence should decrease
    router.update_local(0.9, 0.95, 4);
    router.update_peer([1; 8], 0.1, 0.2, 0, 100);
    router.update_peer([2; 8], 0.05, 0.1, 0, 100);

    let divergent_cadence = router.sharing_cadence();
    assert!(
        divergent_cadence < initial_cadence,
        "High divergence ({}) should decrease cadence from {} to lower, got {}",
        router.collective_divergence(),
        initial_cadence,
        divergent_cadence
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Moral emergency routing
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_moral_emergency_bypasses_bandwidth() {
    let mut router = ConsciousnessAwareRouter::default();
    let classifier = PayloadClassifier::default();

    // Add a high-Phi peer as potential relay
    router.update_local(0.6, 0.7, 2);
    router.update_peer([0xFF; 8], 0.95, 0.98, 4, 100);

    // Signal moral emergency
    router.signal_moral_emergency();

    // Route should be MoralEmergency with the high-Phi peer as relay
    let decision = router.route(PayloadClass::Emergency, 40, 2, &classifier);
    match decision {
        ConsciousRoutingDecision::MoralEmergency {
            tier,
            preferred_relay,
        } => {
            assert_eq!(
                tier,
                symthaea::cognitive_loop::managers::radio_dispatcher::RadioTier::Local
            );
            assert_eq!(preferred_relay, Some([0xFF; 8]));
        }
        other => panic!("Expected MoralEmergency, got {other:?}"),
    }

    // Next route should be normal (one-shot)
    let decision2 = router.route(PayloadClass::Discovery, 64, 1, &classifier);
    assert!(
        matches!(decision2, ConsciousRoutingDecision::Normal(_)),
        "Moral emergency should be one-shot"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Collective immune threat corroboration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_threat_corroboration_across_nodes() {
    let mut node_a = ConsciousnessAwareRouter::default();
    let mut node_b = ConsciousnessAwareRouter::default();

    let threat = ThreatObservation {
        threat_type: 3,
        severity: 0.85,
        agent_hash: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
        signature: [0x42; 32],
        observed_cycle: 100,
        corroboration_count: 0,
    };

    // Node A detects threat
    node_a.record_threat(threat.clone());
    assert_eq!(node_a.threat_count(), 1);
    assert_eq!(node_a.threats()[0].corroboration_count, 0);

    // Node B independently detects same threat → corroboration
    node_b.record_threat(threat.clone());
    // Simulate Node A receiving Node B's threat report
    node_a.record_threat(threat.clone());

    // Should corroborate, not duplicate
    assert_eq!(node_a.threat_count(), 1);
    assert_eq!(node_a.threats()[0].corroboration_count, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dream-consolidated store-and-forward
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_dream_consolidation_on_reconnect() {
    let mut sf = StoreAndForward::default();

    // Go offline
    sf.go_offline(100);
    assert!(sf.is_offline());

    // Accumulate mixed experiences
    for i in 0..20 {
        sf.record(OfflineExperience {
            cycle: 100 + i,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: format!("water-tank-{}", i % 3),
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

    // Reconnect
    let needs_consolidation = sf.go_online(200);
    assert!(
        needs_consolidation,
        "Should need consolidation after 22 experiences"
    );
    assert!(!sf.is_offline());
    assert_eq!(sf.reconnection_count(), 1);

    // Consolidate
    let wisdom = sf.consolidate(200);
    assert_eq!(wisdom.experiences_consolidated, 22);
    assert_eq!(wisdom.offline_duration, 100);
    assert!(wisdom.mean_salience > 0.5);

    // Should have patterns for: sensor trends, consciousness trajectory, threat summary
    assert!(
        wisdom.patterns.len() >= 3,
        "Expected ≥3 patterns (sensor, consciousness, threat), got {}",
        wisdom.patterns.len()
    );

    let pattern_kinds: Vec<&str> = wisdom.patterns.iter().map(|p| p.kind.as_str()).collect();
    assert!(pattern_kinds.contains(&"sensor_trend"));
    assert!(pattern_kinds.contains(&"consciousness_trajectory"));
    assert!(pattern_kinds.contains(&"threat_summary"));

    // Buffer should be empty after consolidation
    assert_eq!(sf.buffer_len(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compressed delta consciousness sharing
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_compressed_delta_consciousness_sharing() {
    // Simulate two nodes sharing BinaryHV consciousness vectors via compressed deltas

    // Node A's initial state
    let mut state_a = [0u8; 2048];
    for i in 0..2048 {
        state_a[i] = (i % 256) as u8;
    }

    // Node A evolves slightly (flip ~5% of bits)
    let mut state_a_next = state_a;
    for i in (0..2048).step_by(20) {
        state_a_next[i] ^= 0xFF;
    }

    // Compress the delta
    let delta = CompressedDelta::from_diff(&state_a, &state_a_next);
    assert!(!delta.is_full, "Should be a delta, not a full vector");
    assert!(
        delta.wire_size() < 400,
        "Delta should compress to <400 bytes, got {}",
        delta.wire_size()
    );

    // Node B applies the delta to reconstruct Node A's new state
    let reconstructed = delta
        .apply(&state_a)
        .expect("Delta application should succeed");
    assert_eq!(
        reconstructed, state_a_next,
        "Reconstructed state should match original"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Discovery beacon consciousness fields
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_beacon_consciousness_roundtrip() {
    let beacon = DiscoveryBeacon {
        node_id: [1; 8],
        capabilities_hash: [2; 8],
        cycle_counter: 999,
        network_health: 0,
        tier_mask: 0x07,
        phi_quantized: DiscoveryBeacon::quantize_phi(0.85),
        governance_tier: 4, // Guardian
    };

    let bytes = beacon.to_bytes();
    assert_eq!(bytes.len(), 24);

    let decoded = DiscoveryBeacon::from_bytes(&bytes);
    let phi = DiscoveryBeacon::dequantize_phi(decoded.phi_quantized);

    assert!(
        (phi - 0.85).abs() < 0.01,
        "Phi roundtrip: expected ~0.85, got {phi}"
    );
    assert_eq!(decoded.governance_tier, 4);
}

#[test]
fn test_beacon_backward_compatibility() {
    // Old nodes send 0 for bytes 22-23 (previously reserved).
    // New code should interpret this as: unknown Phi (0.0), Observer tier (0).
    let mut bytes = [0u8; 24];
    bytes[0..8].copy_from_slice(&[1; 8]); // node_id
    bytes[16..20].copy_from_slice(&42u32.to_le_bytes()); // cycle
    bytes[20] = 0; // health
    bytes[21] = 0x07; // tier_mask
                      // bytes[22] and [23] are 0 (old node didn't set them)

    let decoded = DiscoveryBeacon::from_bytes(&bytes);
    assert_eq!(decoded.phi_quantized, 0);
    assert_eq!(decoded.governance_tier, 0);
    assert_eq!(DiscoveryBeacon::dequantize_phi(0), 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trust decay on suspicious behavior
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_trust_decays_on_phi_jumps() {
    let mut router = ConsciousnessAwareRouter::default();

    // Peer reports stable Phi
    for cycle in 0..10 {
        router.update_peer([1; 8], 0.5, 0.6, 2, cycle);
    }
    let trust_stable = router.peers_by_trust()[0].1;

    // Peer suddenly reports wildly different Phi (suspicious)
    router.update_peer([1; 8], 0.99, 0.6, 2, 10);
    let trust_after_jump = router.peers_by_trust()[0].1;

    assert!(
        trust_after_jump < trust_stable,
        "Trust should decay after suspicious Phi jump: {trust_stable} → {trust_after_jump}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Highest-Phi relay selection
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_highest_phi_relay_selection() {
    let mut router = ConsciousnessAwareRouter::default();

    router.update_peer([1; 8], 0.3, 0.4, 1, 100);
    router.update_peer([2; 8], 0.9, 0.95, 4, 100); // Highest
    router.update_peer([3; 8], 0.6, 0.7, 2, 100);

    let best = router.highest_phi_peer();
    assert_eq!(best, Some([2; 8]), "Should select highest-Phi peer");

    // Sorted list should have [2] first
    let sorted = router.peers_by_trust();
    assert_eq!(sorted[0].0, [2; 8]);
}
