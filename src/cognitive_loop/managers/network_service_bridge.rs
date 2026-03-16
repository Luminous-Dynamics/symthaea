//! # Network Service Bridge — Async→Sync Adapter for SwarmManager
//!
//! Bridges the async [`NetworkService`] (tokio broadcast channels) to the sync
//! cognitive loop via an `mpsc::Sender<SwarmEvent>` channel.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────── Async (tokio) ───────┐     ┌────── Sync (50Hz CLS) ──────┐
//! │                              │     │                              │
//! │  NetworkService              │     │  swarm_event_rx              │
//! │    ├─ peer_event_tx ─────►   │     │    │                        │
//! │    └─ consciousness_tx ──►   │     │    ▼                        │
//! │                              │     │  SwarmManager.inject_event() │
//! │  Hyperfeel                   │ mpsc│    │                        │
//! │    └─ affective_state ───► ─────► │    ▼                        │
//! │                              │     │  process() → SubsystemOutput │
//! │  FederatedAggregator         │     │    │                        │
//! │    └─ round_complete ────►   │     │    ▼                        │
//! │                              │     │  apply_swarm_neuromod()     │
//! └──────────────────────────────┘     └──────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! // On CLS (sync side):
//! let tx = cls.create_swarm_event_channel();
//!
//! // On async side:
//! let bridge = NetworkServiceBridge::spawn(&network_service, tx.clone());
//!
//! // From Hyperfeel (async):
//! let _ = tx.send(SwarmEvent::AffectiveSync { .. });
//!
//! // CLS automatically drains the channel in Phase B.
//! ```

use super::swarm_manager::{
    convert_affective_sync, convert_consciousness_vector, convert_peer_event, SwarmEvent,
    SwarmManager,
};
use crate::swarm::{AffectiveSync, ConsciousnessVector, NetworkService, PeerEvent};
use std::sync::mpsc;
use tokio::sync::broadcast;

/// Cap on events drained per poll cycle to avoid starving the cognitive loop.
/// At 50Hz with 100-peer swarm, bursts rarely exceed ~20 events.
pub(crate) const MAX_EVENTS_PER_POLL: usize = 64;

/// Handle returned by [`NetworkServiceBridge::spawn`].
///
/// Dropping this handle does NOT stop the background task — the task runs
/// until the broadcast channels close (NetworkService shutdown) or the
/// mpsc sender is disconnected (CLS dropped).
pub struct NetworkServiceBridgeHandle {
    /// Cumulative events forwarded (atomic for lock-free reads from sync side).
    total_forwarded: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Cumulative events dropped due to broadcast lag.
    total_lagged: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl NetworkServiceBridgeHandle {
    /// Total events successfully forwarded to the mpsc channel.
    pub fn total_forwarded(&self) -> u64 {
        self.total_forwarded
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total events dropped due to broadcast channel overflow.
    pub fn total_lagged(&self) -> u64 {
        self.total_lagged.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Async→sync bridge that runs as a tokio task.
///
/// Subscribes to [`NetworkService`] broadcast channels, converts events to
/// [`SwarmEvent`]s, and forwards them through an `mpsc::Sender` to the CLS.
pub struct NetworkServiceBridge;

impl NetworkServiceBridge {
    /// Spawn an async task that forwards NetworkService events to the CLS channel.
    ///
    /// The task runs until either:
    /// - All broadcast channels close (NetworkService shutdown)
    /// - The `mpsc::Sender` disconnects (CLS dropped)
    ///
    /// Returns a handle for diagnostic counters.
    pub fn spawn(
        service: &NetworkService,
        tx: mpsc::Sender<SwarmEvent>,
    ) -> NetworkServiceBridgeHandle {
        let mut peer_rx = service.subscribe_peer_events();
        let mut consciousness_rx = service.subscribe_consciousness();
        let forwarded = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let lagged = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

        let fwd = forwarded.clone();
        let lag = lagged.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = peer_rx.recv() => {
                        match result {
                            Ok(event) => {
                                if let Some(swarm_event) = convert_peer_event(&event) {
                                    if tx.send(swarm_event).is_err() {
                                        break; // CLS dropped
                                    }
                                    fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    lagged = n,
                                    "NetworkServiceBridge: peer_rx lagged"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                    result = consciousness_rx.recv() => {
                        match result {
                            Ok((peer_id, cv)) => {
                                let event = convert_consciousness_vector(&peer_id, &cv);
                                if tx.send(event).is_err() {
                                    break; // CLS dropped
                                }
                                fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    lagged = n,
                                    "NetworkServiceBridge: consciousness_rx lagged"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
            tracing::info!("NetworkServiceBridge: task exiting");
        });

        NetworkServiceBridgeHandle {
            total_forwarded: forwarded,
            total_lagged: lagged,
        }
    }
}

/// Convenience function to forward Hyperfeel affective state into the CLS channel.
///
/// Call this from async context after `hyperfeel.receive_peer_state()`:
/// ```rust,ignore
/// forward_affective_state(&tx, peer_id, affect);
/// ```
pub fn forward_affective_state(tx: &mpsc::Sender<SwarmEvent>, peer_id: &str, sync: &AffectiveSync) {
    let event = convert_affective_sync(peer_id, sync);
    let _ = tx.send(event);
}

/// Convenience function to forward a FederatedAggregator round result.
///
/// Call after `aggregator.aggregate()` completes:
/// ```rust,ignore
/// forward_federated_round(&tx, n_contributors, avg_quality, trust_confidence);
/// ```
pub fn forward_federated_round(
    tx: &mpsc::Sender<SwarmEvent>,
    n_contributors: usize,
    avg_quality: f64,
    trust_confidence: f64,
) {
    let event = SwarmEvent::FederatedRound {
        n_contributors,
        avg_quality: avg_quality.clamp(0.0, 1.0),
        trust_confidence: trust_confidence.clamp(0.0, 1.0),
    };
    let _ = tx.send(event);
}

/// Synchronous poll helper for draining the CLS-side receiver.
///
/// Used internally by `cycle_phase_dynamics.rs`. Returns the number of events
/// drained into the SwarmManager.
pub fn drain_swarm_channel(rx: &mpsc::Receiver<SwarmEvent>, manager: &mut SwarmManager) -> usize {
    let mut count = 0;
    for _ in 0..MAX_EVENTS_PER_POLL {
        match rx.try_recv() {
            Ok(event) => {
                manager.inject_event(event);
                count += 1;
            }
            Err(_) => break,
        }
    }
    count
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::{ConsciousnessVector, PeerEvent, PeerInfo};

    fn make_peer_info(id: &str) -> PeerInfo {
        PeerInfo::new(id)
    }

    fn make_cv(phi: f64) -> ConsciousnessVector {
        ConsciousnessVector::new(vec![0.0; 64], phi)
    }

    #[test]
    fn test_drain_peer_events() {
        let (tx, rx) = mpsc::channel();
        let mut manager = SwarmManager::default();

        // Simulate NetworkService sending a PeerConnected
        let peer = make_peer_info("peer-1");
        if let Some(event) = convert_peer_event(&PeerEvent::Connected(peer)) {
            tx.send(event).unwrap();
        }

        let n = drain_swarm_channel(&rx, &mut manager);
        assert_eq!(n, 1);
        // connected_peers is updated during process(), not inject_event()
        // Just verify the event was drained from the channel
        assert!(
            rx.try_recv().is_err(),
            "Channel should be empty after drain"
        );
    }

    #[test]
    fn test_drain_consciousness_updates() {
        let (tx, rx) = mpsc::channel();
        let mut manager = SwarmManager::default();

        let event_a = convert_consciousness_vector("peer-A", &make_cv(0.7));
        let event_b = convert_consciousness_vector("peer-B", &make_cv(0.3));
        tx.send(event_a).unwrap();
        tx.send(event_b).unwrap();

        let n = drain_swarm_channel(&rx, &mut manager);
        assert_eq!(n, 2);
    }

    #[test]
    fn test_empty_channel_returns_zero() {
        let (_tx, rx) = mpsc::channel::<SwarmEvent>();
        let mut manager = SwarmManager::default();

        let n = drain_swarm_channel(&rx, &mut manager);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_forward_affective_state() {
        let (tx, rx) = mpsc::channel();
        let sync = AffectiveSync {
            valence: 0.5,
            arousal: 0.7,
            dominance: 0.3,
            timestamp_ms: 0,
            sequence: 0,
        };
        forward_affective_state(&tx, "peer-1", &sync);

        let event = rx.try_recv().unwrap();
        match event {
            SwarmEvent::AffectiveSync {
                peer_id, valence, ..
            } => {
                assert_eq!(peer_id, "peer-1");
                assert!((valence - 0.5).abs() < 0.01);
            }
            _ => panic!("Expected AffectiveSync"),
        }
    }

    #[test]
    fn test_forward_federated_round() {
        let (tx, rx) = mpsc::channel();
        forward_federated_round(&tx, 5, 0.8, 0.9);

        let event = rx.try_recv().unwrap();
        match event {
            SwarmEvent::FederatedRound {
                n_contributors,
                avg_quality,
                trust_confidence,
            } => {
                assert_eq!(n_contributors, 5);
                assert!((avg_quality - 0.8).abs() < 0.01);
                assert!((trust_confidence - 0.9).abs() < 0.01);
            }
            _ => panic!("Expected FederatedRound"),
        }
    }

    #[test]
    fn test_max_events_cap() {
        let (tx, rx) = mpsc::channel();
        let mut manager = SwarmManager::default();

        // Send more than MAX_EVENTS_PER_POLL events
        for i in 0..80 {
            let event = SwarmEvent::PeerJoined {
                peer_id: format!("p-{i}"),
                trust_level: 0.5,
            };
            tx.send(event).unwrap();
        }

        let n = drain_swarm_channel(&rx, &mut manager);
        assert!(
            n <= MAX_EVENTS_PER_POLL,
            "Should cap at MAX_EVENTS_PER_POLL={}, got {}",
            MAX_EVENTS_PER_POLL,
            n
        );
    }

    #[test]
    fn test_federated_round_clamps() {
        let (tx, rx) = mpsc::channel();
        forward_federated_round(&tx, 0, 1.5, -0.3);

        let event = rx.try_recv().unwrap();
        match event {
            SwarmEvent::FederatedRound {
                avg_quality,
                trust_confidence,
                ..
            } => {
                assert!((avg_quality - 1.0).abs() < 0.01, "Should clamp to 1.0");
                assert!((trust_confidence - 0.0).abs() < 0.01, "Should clamp to 0.0");
            }
            _ => panic!("Expected FederatedRound"),
        }
    }

    #[tokio::test]
    async fn test_spawn_forwards_events() {
        let (tx, rx) = mpsc::channel();

        // Create broadcast channels to simulate NetworkService
        let (peer_tx, _) = tokio::sync::broadcast::channel::<PeerEvent>(16);
        let (cons_tx, _) = tokio::sync::broadcast::channel::<(String, ConsciousnessVector)>(16);

        // We can't use NetworkServiceBridge::spawn directly without a real
        // NetworkService, but we can test the channel flow end-to-end
        let peer_rx = peer_tx.subscribe();
        let consciousness_rx = cons_tx.subscribe();

        let forwarded = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let lagged = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let fwd = forwarded.clone();

        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut peer_rx = peer_rx;
            let mut consciousness_rx = consciousness_rx;
            loop {
                tokio::select! {
                    result = peer_rx.recv() => {
                        match result {
                            Ok(event) => {
                                if let Some(swarm_event) = convert_peer_event(&event) {
                                    if tx_clone.send(swarm_event).is_err() { break; }
                                    fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    result = consciousness_rx.recv() => {
                        match result {
                            Ok((peer_id, cv)) => {
                                let event = convert_consciousness_vector(&peer_id, &cv);
                                if tx_clone.send(event).is_err() { break; }
                                fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(_) => break,
                        }
                    }
                }
            }
        });

        // Send events through broadcast channels
        peer_tx
            .send(PeerEvent::Connected(make_peer_info("p1")))
            .unwrap();
        cons_tx.send(("p2".into(), make_cv(0.6))).unwrap();

        // Give async task time to forward
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Drain on sync side
        let mut manager = SwarmManager::default();
        let n = drain_swarm_channel(&rx, &mut manager);
        assert_eq!(n, 2, "Should have forwarded peer + consciousness events");
        assert_eq!(forwarded.load(std::sync::atomic::Ordering::Relaxed), 2);
    }
}
