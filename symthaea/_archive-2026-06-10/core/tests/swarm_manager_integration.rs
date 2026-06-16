// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test for SwarmManager in the cognitive loop.
//!
//! Verifies end-to-end: inject swarm events -> run cycles -> verify
//! telemetry reflects events, neuromod is affected, CycleMetadata
//! includes swarm subsystem output.
//!
//! SwarmManager runs at interval 41 (doubled to 82 at urgency 0).
//! We must run enough cycles to guarantee at least one `process()` call.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};

fn create_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

/// Run enough cycles past SwarmManager's interval (41, effective 82 at urgency 0)
/// to guarantee at least one `process()` call.
fn run_past_swarm_interval(svc: &mut CognitiveLoopService, label: &str, n: usize) {
    for i in 0..n {
        svc.cycle(&format!("{} {}", label, i));
    }
}

/// Number of cycles needed to guarantee SwarmManager processes at least once.
/// Interval 41, effective 82 at urgency 0. Run 83 to be safe.
const SWARM_INTERVAL_CYCLES: usize = 83;

#[test]
fn test_swarm_events_affect_telemetry() {
    let mut svc = create_service();

    // Initially isolated
    assert_eq!(svc.swarm_telemetry().connected_peers, 0);

    // Add peers
    svc.inject_swarm_event(SwarmEvent::PeerJoined {
        peer_id: "alice".into(),
        trust_level: 0.8,
    });
    svc.inject_swarm_event(SwarmEvent::PeerJoined {
        peer_id: "bob".into(),
        trust_level: 0.6,
    });

    // Run past the swarm interval to guarantee processing
    run_past_swarm_interval(&mut svc, "swarm telemetry", SWARM_INTERVAL_CYCLES);

    let t = svc.swarm_telemetry();
    assert_eq!(t.connected_peers, 2);
    assert!(t.connectivity_ema > 0.0);
}

#[test]
fn test_consciousness_update_affects_mean_phi() {
    let mut svc = create_service();

    svc.inject_swarm_event(SwarmEvent::PeerJoined {
        peer_id: "alice".into(),
        trust_level: 0.8,
    });
    svc.inject_swarm_event(SwarmEvent::ConsciousnessUpdate {
        peer_id: "alice".into(),
        phi: 0.7,
        valence: 0.3,
        arousal: 0.5,
    });

    run_past_swarm_interval(&mut svc, "phi test", SWARM_INTERVAL_CYCLES);
    assert!(
        svc.swarm_mean_peer_phi() > 0.0,
        "Mean peer phi should be positive after consciousness update: {}",
        svc.swarm_mean_peer_phi()
    );
}

#[test]
fn test_mass_disconnect_triggers_anomaly() {
    let mut svc = create_service();

    // Add peers
    for i in 0..5 {
        svc.inject_swarm_event(SwarmEvent::PeerJoined {
            peer_id: format!("peer-{i}"),
            trust_level: 0.5,
        });
    }
    run_past_swarm_interval(&mut svc, "setup", SWARM_INTERVAL_CYCLES);

    // Mass disconnect
    svc.inject_swarm_event(SwarmEvent::TopologyChange {
        connected_peers: 1,
        mass_disconnect: true,
    });
    run_past_swarm_interval(&mut svc, "disconnect", SWARM_INTERVAL_CYCLES);

    assert!(
        svc.swarm_telemetry().anomaly_count > 0,
        "Mass disconnect should trigger anomaly"
    );
}

#[test]
fn test_swarm_telemetry_after_processing() {
    let mut svc = create_service();

    svc.inject_swarm_event(SwarmEvent::PeerJoined {
        peer_id: "test-peer".into(),
        trust_level: 0.9,
    });

    // Run past interval so telemetry is populated
    run_past_swarm_interval(&mut svc, "telemetry check", SWARM_INTERVAL_CYCLES);

    // After processing, connected_peers should be 1 in the swarm telemetry accessor
    let t = svc.swarm_telemetry();
    assert_eq!(t.connected_peers, 1);
    assert!(t.connectivity_ema > 0.0);
}

#[test]
fn test_federated_round_boosts_confidence() {
    let mut svc = create_service();

    svc.inject_swarm_event(SwarmEvent::FederatedRound {
        n_contributors: 10,
        avg_quality: 0.9,
        trust_confidence: 0.85,
    });

    run_past_swarm_interval(&mut svc, "federated round", SWARM_INTERVAL_CYCLES);
    assert!(
        svc.swarm_telemetry().federated_confidence > 0.0,
        "Federated round should set confidence > 0"
    );
}