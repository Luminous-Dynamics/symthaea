//! Integration test for SwarmManager in the cognitive loop.
//!
//! Verifies end-to-end: inject swarm events -> run cycles -> verify
//! telemetry reflects events, neuromod is affected, CycleMetadata
//! includes swarm subsystem output.

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};

fn create_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

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

    // Run a cycle to process
    svc.cycle("swarm integration test");

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

    svc.cycle("phi test");
    assert!(svc.swarm_mean_peer_phi() > 0.0);
}

#[test]
fn test_mass_disconnect_triggers_anomaly() {
    let mut svc = create_service();

    // Add peers then mass disconnect
    for i in 0..5 {
        svc.inject_swarm_event(SwarmEvent::PeerJoined {
            peer_id: format!("peer-{i}"),
            trust_level: 0.5,
        });
    }
    svc.cycle("setup");

    svc.inject_swarm_event(SwarmEvent::TopologyChange {
        connected_peers: 1,
        mass_disconnect: true,
    });
    svc.cycle("disconnect");

    assert!(svc.swarm_telemetry().anomaly_count > 0);
}

#[test]
fn test_swarm_telemetry_in_metadata() {
    let mut svc = create_service();

    svc.inject_swarm_event(SwarmEvent::PeerJoined {
        peer_id: "test-peer".into(),
        trust_level: 0.9,
    });

    let output = svc.cycle("metadata test");
    assert_eq!(output.metadata.swarm_connected_peers, 1);
}

#[test]
fn test_federated_round_boosts_confidence() {
    let mut svc = create_service();

    svc.inject_swarm_event(SwarmEvent::FederatedRound {
        n_contributors: 10,
        avg_quality: 0.9,
        trust_confidence: 0.85,
    });

    svc.cycle("federated round");
    assert!(svc.swarm_telemetry().federated_confidence > 0.0);
}
