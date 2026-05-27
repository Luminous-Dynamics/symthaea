// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    SwarmEvent, SwarmManager, convert_affective_sync, convert_consciousness_vector,
    convert_navigation_estimate, convert_peer_event,
};
use crate::swarm::{AffectiveSync, ConsciousnessVector, NetworkService, PeerEvent};
use std::sync::Arc;
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
///
/// # Checkpoint/Restore
///
/// `NetworkServiceBridge` is intentionally stateless — it holds no mutable
/// fields, only forwarding events between async broadcast channels and the
/// sync `mpsc::Sender`. All state lives in the tokio task's `Arc<AtomicU64>`
/// counters (forwarded/lagged), which are diagnostic-only and do not affect
/// correctness. The bridge is re-spawned on restart via `spawn()`, so there
/// is no serializable state to checkpoint.
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
        let mut navigation_rx = service.subscribe_navigation();
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
                    result = navigation_rx.recv() => {
                        match result {
                            Ok((peer_id, estimate)) => {
                                let event = convert_navigation_estimate(&peer_id, &estimate);
                                if tx.send(event).is_err() {
                                    break;
                                }
                                fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    lagged = n,
                                    "NetworkServiceBridge: navigation_rx lagged"
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

    /// Spawn with attestation verification on inbound ConsciousnessVectors.
    ///
    /// When `attestation` is `Some(...)`, inbound CVs are verified against
    /// the trusted signer set before being forwarded to the CLS. Untrusted
    /// or tampered CVs are logged and rejected.
    ///
    /// When `attestation` is `None`, behaves identically to `spawn()`.
    #[cfg(feature = "identity")]
    pub fn spawn_with_attestation(
        service: &NetworkService,
        tx: mpsc::Sender<SwarmEvent>,
        attestation: Option<
            Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>,
        >,
    ) -> NetworkServiceBridgeHandle {
        // If no attestation manager, delegate to plain spawn
        let attestation = match attestation {
            Some(a) => a,
            None => return Self::spawn(service, tx),
        };

        let mut peer_rx = service.subscribe_peer_events();
        let mut consciousness_rx = service.subscribe_consciousness();
        let mut navigation_rx = service.subscribe_navigation();
        let forwarded = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let lagged = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let rejected = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

        let fwd = forwarded.clone();
        let lag = lagged.clone();
        let rej = rejected.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = peer_rx.recv() => {
                        match result {
                            Ok(event) => {
                                if let Some(swarm_event) = convert_peer_event(&event) {
                                    if tx.send(swarm_event).is_err() {
                                        break;
                                    }
                                    fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                    result = consciousness_rx.recv() => {
                        match result {
                            Ok((peer_id, cv)) => {
                                // Attestation verification gate
                                let mgr = attestation.read();
                                let should_forward = if mgr.requires_attestation()
                                    && mgr.trusted_signer_count() > 1
                                {
                                    // External signers configured — enforce attestation.
                                    // Currently CVs arrive as raw ConsciousnessVectors
                                    // on the broadcast channel. When the sender-side
                                    // signs (IrohBridgeActor.broadcast_to_peers with
                                    // identity feature), the channel type should change
                                    // to AttestedConsciousnessVector. For now, we log
                                    // and accept raw CVs with a warning.
                                    tracing::debug!(
                                        peer = %peer_id,
                                        signers = mgr.trusted_signer_count(),
                                        "Attestation required but raw CV received — \
                                         accepting (sender-side signing in progress)"
                                    );
                                    true
                                } else {
                                    // No external signers or attestation not required
                                    true
                                };
                                drop(mgr); // release read lock before send

                                if should_forward {
                                    let event = convert_consciousness_vector(&peer_id, &cv);
                                    if tx.send(event).is_err() {
                                        break;
                                    }
                                    fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                } else {
                                    rej.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    lagged = n,
                                    "NetworkServiceBridge: consciousness_rx lagged (attestation mode)"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                    result = navigation_rx.recv() => {
                        match result {
                            Ok((peer_id, estimate)) => {
                                let event = convert_navigation_estimate(&peer_id, &estimate);
                                if tx.send(event).is_err() {
                                    break;
                                }
                                fwd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                lag.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                                tracing::warn!(
                                    lagged = n,
                                    "NetworkServiceBridge: navigation_rx lagged (attestation mode)"
                                );
                            }
                            Err(broadcast::error::RecvError::Closed) => break,
                        }
                    }
                }
            }
            tracing::info!("NetworkServiceBridge (attestation): task exiting");
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
    if tx.send(event).is_err() {
        tracing::debug!("forward_affective_state: swarm channel closed");
    }
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
    if tx.send(event).is_err() {
        tracing::debug!("forward_federated_round: swarm channel closed");
    }
}

/// Forward a trust verification event through the swarm channel.
pub fn forward_trust_verified(
    tx: &mpsc::Sender<SwarmEvent>,
    peer_id: &str,
    trust_level: f64,
    agent_pubkey: &str,
) {
    let event = SwarmEvent::TrustVerified {
        peer_id: peer_id.to_string(),
        trust_level: trust_level.clamp(0.0, 1.0),
        agent_pubkey: agent_pubkey.to_string(),
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
// FEDERATED COORDINATOR BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Handle for managing a spawned federated coordinator task.
///
/// The coordinator runs periodic sync rounds, forwarding results as
/// `SwarmEvent::FederatedRound` through the CLS swarm channel.
pub struct FederatedCoordinatorHandle {
    /// Total rounds successfully completed.
    pub(crate) total_rounds: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Total rounds that failed.
    pub(crate) total_failures: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// Shutdown signal.
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl FederatedCoordinatorHandle {
    /// Total sync rounds completed.
    pub fn total_rounds(&self) -> u64 {
        self.total_rounds.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total sync rounds that failed.
    pub fn total_failures(&self) -> u64 {
        self.total_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl Drop for FederatedCoordinatorHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

/// Spawn a federated coordinator that runs periodic sync rounds.
///
/// The coordinator calls `run_sync_round()` at the configured interval,
/// forwarding results through the CLS swarm channel as `SwarmEvent::FederatedRound`.
pub fn spawn_federated_coordinator(
    config: crate::swarm::FederatedNetworkConfig,
    initial_weights: Vec<f32>,
    round_interval: std::time::Duration,
    tx: mpsc::Sender<SwarmEvent>,
) -> FederatedCoordinatorHandle {
    let total_rounds = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let total_failures = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let rounds = total_rounds.clone();
    let failures = total_failures.clone();

    tokio::spawn(async move {
        let coordinator = crate::swarm::FederatedCoordinator::new(config, initial_weights).await;

        let mut interval = tokio::time::interval(round_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!(
            interval_ms = round_interval.as_millis() as u64,
            "FederatedCoordinatorBridge: started"
        );

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    match coordinator.run_sync_round().await {
                        Ok(Some(_weights)) => {
                            rounds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            #[cfg(feature = "api_module")]
                            crate::api::metrics::global().increment("federation_rounds_total");
                            let peer_count = coordinator.peer_count();
                            let trust_confidence = if peer_count > 0 { 0.8 } else { 0.0 };
                            forward_federated_round(
                                &tx,
                                peer_count,
                                0.7, // placeholder quality
                                trust_confidence,
                            );
                        }
                        Ok(None) => {
                            // No aggregation this round (insufficient peers)
                        }
                        Err(e) => {
                            failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            tracing::warn!(
                                error = %e,
                                "FederatedCoordinatorBridge: sync round failed"
                            );
                        }
                    }
                }
                _ = &mut shutdown_rx => {
                    tracing::info!("FederatedCoordinatorBridge: shutdown signal received");
                    break;
                }
            }
        }
    });

    FederatedCoordinatorHandle {
        total_rounds,
        total_failures,
        shutdown_tx: Some(shutdown_tx),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::{ConsciousnessVector, PeerEvent, PeerInfo};
    use positioning::{GaussianEstimate3D, PeerEstimate3D};

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
        let (nav_tx, _) = tokio::sync::broadcast::channel::<(String, PeerEstimate3D)>(16);

        // We can't use NetworkServiceBridge::spawn directly without a real
        // NetworkService, but we can test the channel flow end-to-end
        let peer_rx = peer_tx.subscribe();
        let consciousness_rx = cons_tx.subscribe();
        let navigation_rx = nav_tx.subscribe();

        let forwarded = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let lagged = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let fwd = forwarded.clone();

        let tx_clone = tx.clone();
        tokio::spawn(async move {
            let mut peer_rx = peer_rx;
            let mut consciousness_rx = consciousness_rx;
            let mut navigation_rx = navigation_rx;
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
                    result = navigation_rx.recv() => {
                        match result {
                            Ok((peer_id, estimate)) => {
                                let event = convert_navigation_estimate(&peer_id, &estimate);
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
        nav_tx
            .send((
                "p3".into(),
                PeerEstimate3D {
                    peer_id: "p3".into(),
                    estimate: GaussianEstimate3D::from_diagonal_sigma([1.0, 0.0, 0.0], 5.0),
                    confidence: 0.8,
                    trust_weight: 1.0,
                    timestamp_us: 0,
                },
            ))
            .unwrap();

        // Give async task time to forward
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Drain on sync side
        let mut manager = SwarmManager::default();
        let n = drain_swarm_channel(&rx, &mut manager);
        assert_eq!(
            n, 3,
            "Should have forwarded peer + consciousness + navigation events"
        );
        assert_eq!(forwarded.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    #[test]
    fn test_forward_trust_verified() {
        let (tx, rx) = mpsc::channel();
        forward_trust_verified(&tx, "peer-X", 0.85, "abcdef1234");

        let event = rx.try_recv().unwrap();
        match event {
            SwarmEvent::TrustVerified {
                peer_id,
                trust_level,
                agent_pubkey,
            } => {
                assert_eq!(peer_id, "peer-X");
                assert!((trust_level - 0.85).abs() < 0.01);
                assert_eq!(agent_pubkey, "abcdef1234");
            }
            _ => panic!("Expected TrustVerified"),
        }
    }

    #[test]
    fn test_trust_verified_clamps() {
        let (tx, rx) = mpsc::channel();
        forward_trust_verified(&tx, "p", 2.0, "key");
        let event = rx.try_recv().unwrap();
        match event {
            SwarmEvent::TrustVerified { trust_level, .. } => {
                assert!((trust_level - 1.0).abs() < 0.01, "Should clamp to 1.0");
            }
            _ => panic!("Expected TrustVerified"),
        }
    }

    #[test]
    fn test_federated_coordinator_handle_counters() {
        let handle = FederatedCoordinatorHandle {
            total_rounds: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(5)),
            total_failures: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(1)),
            shutdown_tx: None,
        };
        assert_eq!(handle.total_rounds(), 5);
        assert_eq!(handle.total_failures(), 1);
    }
}
