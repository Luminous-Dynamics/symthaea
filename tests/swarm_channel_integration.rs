// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the swarm event channel pipeline.
//!
//! Verifies: mpsc::Sender → CLS.swarm_event_rx → drain → SwarmManager → telemetry
//!
//! SwarmManager runs at interval 41 (effective 82 at urgency 0).
//! We must run ≥83 cycles to guarantee at least one `process()` call.

use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, SwarmEvent, forward_affective_state,
    forward_federated_round,
};

fn make_cls() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

/// Run enough cycles past SwarmManager's effective interval.
const SWARM_CYCLES: usize = 83;

fn run_cycles(svc: &mut CognitiveLoopService, n: usize, label: &str) {
    for i in 0..n {
        svc.cycle(&format!("{label} {i}"));
    }
}

#[test]
fn channel_events_reach_swarm_manager() {
    let mut cls = make_cls();
    let tx = cls.create_swarm_event_channel();

    // Send events through the channel
    tx.send(SwarmEvent::PeerJoined {
        peer_id: "peer-alpha".into(),
        trust_level: 0.8,
    })
    .unwrap();
    tx.send(SwarmEvent::ConsciousnessUpdate {
        peer_id: "peer-alpha".into(),
        phi: 0.6,
        valence: 0.2,
        arousal: 0.4,
    })
    .unwrap();

    // Run past swarm interval so Phase B drains channel + SwarmManager processes
    run_cycles(&mut cls, SWARM_CYCLES, "channel");

    let telem = cls.swarm_telemetry();
    assert_eq!(telem.connected_peers, 1, "Peer should be connected");
    assert!(
        telem.mean_peer_phi > 0.0,
        "Mean peer Φ should be > 0 after consciousness update"
    );
}

#[test]
fn channel_affective_sync_reaches_telemetry() {
    let mut cls = make_cls();
    let tx = cls.create_swarm_event_channel();

    // Join a peer first
    tx.send(SwarmEvent::PeerJoined {
        peer_id: "peer-beta".into(),
        trust_level: 0.9,
    })
    .unwrap();

    // Send affective sync via convenience function
    let sync = symthaea::swarm::AffectiveSync {
        valence: 0.7,
        arousal: 0.8,
        dominance: 0.5,
        timestamp_ms: 0,
        sequence: 0,
    };
    forward_affective_state(&tx, "peer-beta", &sync);

    run_cycles(&mut cls, SWARM_CYCLES, "affective");

    let telem = cls.swarm_telemetry();
    assert!(
        telem.affective_contagion >= 0.0,
        "Affective contagion should be non-negative"
    );
}

#[test]
fn channel_federated_round_reaches_telemetry() {
    let mut cls = make_cls();
    let tx = cls.create_swarm_event_channel();

    // Send federated round via convenience function
    forward_federated_round(&tx, 8, 0.85, 0.92);

    run_cycles(&mut cls, SWARM_CYCLES, "federated");

    let telem = cls.swarm_telemetry();
    assert!(
        telem.federated_confidence > 0.0,
        "Federated confidence should be > 0 after round"
    );
}

#[test]
fn channel_mass_disconnect_triggers_anomaly() {
    let mut cls = make_cls();
    let tx = cls.create_swarm_event_channel();

    // Connect several peers
    for i in 0..5 {
        tx.send(SwarmEvent::PeerJoined {
            peer_id: format!("peer-{i}"),
            trust_level: 0.7,
        })
        .unwrap();
    }

    // Process joins
    run_cycles(&mut cls, SWARM_CYCLES, "join");

    // Mass disconnect
    tx.send(SwarmEvent::TopologyChange {
        connected_peers: 1,
        mass_disconnect: true,
    })
    .unwrap();

    run_cycles(&mut cls, SWARM_CYCLES, "disconnect");

    let telem = cls.swarm_telemetry();
    assert!(
        telem.anomaly_count > 0,
        "Mass disconnect should trigger anomaly"
    );
}

#[test]
fn no_channel_runs_fine() {
    // CLS without a channel should work normally (no events, zero telemetry)
    let mut cls = make_cls();

    run_cycles(&mut cls, SWARM_CYCLES, "no-channel");

    let telem = cls.swarm_telemetry();
    assert_eq!(telem.connected_peers, 0);
    assert!((telem.mean_peer_phi - 0.0).abs() < f64::EPSILON);
}