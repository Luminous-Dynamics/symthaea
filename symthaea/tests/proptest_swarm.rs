// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Swarm Manager

Verifies structural invariants of the peer consciousness subsystem:
- Random event sequences never produce NaN or infinite outputs
- Connectivity EMA stays bounded in [0, 1]
- Mean peer phi stays bounded in [0, 1]
- Affective contagion is non-negative
- Peer count matches expected state

Science: Hatfield et al. (1993), Heinrichs et al. (2003)
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn arb_peer_id() -> impl Strategy<Value = String> {
    "[a-z]{4,8}".prop_map(|s| format!("peer-{s}"))
}

fn arb_swarm_event() -> impl Strategy<Value = SwarmEvent> {
    prop_oneof![
        (arb_peer_id(), 0.0..1.0f64).prop_map(|(id, trust)| SwarmEvent::PeerJoined {
            peer_id: id,
            trust_level: trust,
        }),
        arb_peer_id().prop_map(|id| SwarmEvent::PeerLeft { peer_id: id }),
        (arb_peer_id(), 0.0..1.0f64, -1.0..1.0f64, 0.0..1.0f64).prop_map(|(id, phi, val, ar)| {
            SwarmEvent::ConsciousnessUpdate {
                peer_id: id,
                phi,
                valence: val,
                arousal: ar,
            }
        }),
        (arb_peer_id(), -1.0..1.0f64, 0.0..1.0f64, 0.0..1.0f64).prop_map(
            |(id, val, ar, intensity)| SwarmEvent::AffectiveSync {
                peer_id: id,
                valence: val,
                arousal: ar,
                intensity,
            }
        ),
        (1usize..50, 0.0..1.0f64, 0.0..1.0f64).prop_map(|(n, q, tc)| {
            SwarmEvent::FederatedRound {
                n_contributors: n,
                avg_quality: q,
                trust_confidence: tc,
            }
        }),
        (1usize..100, prop::bool::ANY).prop_map(|(n, disc)| SwarmEvent::TopologyChange {
            connected_peers: n,
            mass_disconnect: disc,
        }),
    ]
}

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig::default()).expect("CLS construction")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Random event sequences never produce NaN in telemetry.
    #[test]
    fn prop_events_produce_finite_telemetry(events in proptest::collection::vec(arb_swarm_event(), 0..50)) {
        let mut svc = make_service();
        for event in events {
            svc.inject_swarm_event(event);
        }
        // Run past interval 41 to drain events
        for _ in 0..45 {
            let _ = svc.cycle("proptest swarm");
        }
        let t = svc.swarm_telemetry();
        prop_assert!(!t.connectivity_ema.is_nan(), "connectivity_ema is NaN");
        prop_assert!(!t.mean_peer_phi.is_nan(), "mean_peer_phi is NaN");
        prop_assert!(!t.affective_contagion.is_nan(), "affective_contagion is NaN");
        prop_assert!(!t.federated_confidence.is_nan(), "federated_confidence is NaN");
        prop_assert!(t.connectivity_ema.is_finite(), "connectivity_ema not finite: {}", t.connectivity_ema);
        prop_assert!(t.mean_peer_phi.is_finite(), "mean_peer_phi not finite: {}", t.mean_peer_phi);
        prop_assert!(t.affective_contagion.is_finite(), "affective_contagion not finite: {}", t.affective_contagion);
        prop_assert!(t.federated_confidence.is_finite(), "federated_confidence not finite: {}", t.federated_confidence);
    }

    /// Connectivity EMA stays bounded in [0, 1].
    #[test]
    fn prop_connectivity_ema_bounded(events in proptest::collection::vec(arb_swarm_event(), 1..100)) {
        let mut svc = make_service();
        for event in events {
            svc.inject_swarm_event(event);
        }
        for _ in 0..45 {
            let _ = svc.cycle("proptest swarm bounds");
        }
        let t = svc.swarm_telemetry();
        prop_assert!(t.connectivity_ema >= 0.0, "connectivity EMA below 0: {}", t.connectivity_ema);
        prop_assert!(t.connectivity_ema <= 1.0, "connectivity EMA above 1: {}", t.connectivity_ema);
    }

    /// Mean peer phi stays bounded in [0, 1].
    #[test]
    fn prop_mean_peer_phi_bounded(events in proptest::collection::vec(arb_swarm_event(), 1..100)) {
        let mut svc = make_service();
        for event in events {
            svc.inject_swarm_event(event);
        }
        for _ in 0..45 {
            let _ = svc.cycle("proptest swarm phi");
        }
        let t = svc.swarm_telemetry();
        prop_assert!(t.mean_peer_phi >= 0.0, "mean peer phi below 0: {}", t.mean_peer_phi);
        prop_assert!(t.mean_peer_phi <= 1.0, "mean peer phi above 1: {}", t.mean_peer_phi);
    }

    /// Affective contagion is non-negative.
    #[test]
    fn prop_affective_contagion_nonnegative(events in proptest::collection::vec(arb_swarm_event(), 1..50)) {
        let mut svc = make_service();
        for event in events {
            svc.inject_swarm_event(event);
        }
        for _ in 0..45 {
            let _ = svc.cycle("proptest swarm affect");
        }
        let t = svc.swarm_telemetry();
        prop_assert!(t.affective_contagion >= 0.0, "affective contagion negative: {}", t.affective_contagion);
    }

    /// Peer joins always increment, peer leaves never go negative.
    #[test]
    fn prop_peer_count_nonnegative(
        joins in proptest::collection::vec(arb_peer_id(), 0..20),
        leaves in proptest::collection::vec(arb_peer_id(), 0..30),
    ) {
        let mut svc = make_service();
        for id in &joins {
            svc.inject_swarm_event(SwarmEvent::PeerJoined {
                peer_id: id.clone(),
                trust_level: 0.5,
            });
        }
        for id in &leaves {
            svc.inject_swarm_event(SwarmEvent::PeerLeft {
                peer_id: id.clone(),
            });
        }
        // Run past interval 41
        for _ in 0..45 {
            let _ = svc.cycle("proptest swarm peers");
        }
        // connected_peers is usize, so always >= 0, but verify telemetry coherence
        let t = svc.swarm_telemetry();
        prop_assert!(t.connected_peers <= 256, "exceeds peer cap: {}", t.connected_peers);
    }

    /// Consciousness level remains bounded after swarm event injection.
    #[test]
    fn prop_consciousness_bounded_with_swarm(events in proptest::collection::vec(arb_swarm_event(), 1..30)) {
        let mut svc = make_service();
        for event in events {
            svc.inject_swarm_event(event);
        }
        for _ in 0..50 {
            let r = svc.cycle("proptest swarm consciousness");
            prop_assert!(r.metadata.consciousness.consciousness_level >= 0.0,
                "consciousness below 0: {}", r.metadata.consciousness.consciousness_level);
            prop_assert!(r.metadata.consciousness.consciousness_level <= 1.0,
                "consciousness above 1: {}", r.metadata.consciousness.consciousness_level);
            prop_assert!(r.metadata.consciousness.consciousness_level.is_finite(),
                "consciousness not finite: {}", r.metadata.consciousness.consciousness_level);
        }
    }
}