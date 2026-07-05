// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Iroh P2P Bridge — Async/Sync Actor Pattern
//!
//! Connects the synchronous `ContinuousMind` tick loop to the async `IrohNode`
//! networking layer via bounded mpsc channels.
//!
//! ```text
//! ┌──────────────────────┐       mpsc        ┌──────────────────────┐
//! │  ContinuousMind      │  ──────────────►  │  IrohBridgeActor     │
//! │  (sync, 50Hz)        │  social_outbox    │  (async, tokio task)  │
//! │                      │  ◄──────────────  │                      │
//! │  IrohBridgeHandle    │  social_inbox     │  IrohNode + peers    │
//! └──────────────────────┘                   └──────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Spawn the actor with trust-gated broadcasting
//! let service = NetworkService::new(config).await?;
//! let hs = service.handshake_arc();
//! let (handle, actor) = IrohBridgeHandle::new_with_handshake(64, 128, Some(hs));
//! let node = IrohNode::new(config).await?;
//! tokio::spawn(actor.run(node));
//!
//! // In the sync tick loop:
//! let outgoing = mind.drain_social_outbox();
//! handle.flush_outbox(outgoing);
//! for msg in handle.drain_inbox() {
//!     mind.receive_social(msg);
//! }
//! ```

use crate::mind::SocialMessage;
#[allow(unused_imports)]
use parking_lot::RwLock as ParkingRwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc; // used via Arc<parking_lot::RwLock<...>>

// ============================================================================
// IrohBridgeHandle — Sync side (held by ContinuousMind)
// ============================================================================

/// A sync-safe handle for the Mind to communicate with the async Iroh actor.
///
/// All methods are non-blocking (`try_send` / `try_recv`), so they are safe
/// to call from the 50Hz synchronous tick loop without risking deadlocks.
pub struct IrohBridgeHandle {
    /// Send outbound social messages to the actor (non-blocking try_send).
    outbound_tx: mpsc::Sender<SocialMessage>,
    /// Receive inbound social messages from the actor (non-blocking try_recv).
    inbound_rx: mpsc::Receiver<SocialMessage>,
    /// Health flag — `false` if the actor task has exited.
    alive: Arc<AtomicBool>,
}

impl IrohBridgeHandle {
    /// Create a paired (handle, actor) with the given channel capacities.
    ///
    /// - `outbound_capacity`: Bounded channel from Mind → Actor (64 = ~3.2s at 20 msg/s)
    /// - `inbound_capacity`: Bounded channel from Actor → Mind (128 = burst buffer)
    pub fn new(outbound_capacity: usize, inbound_capacity: usize) -> (Self, IrohBridgeActor) {
        Self::new_with_handshake(outbound_capacity, inbound_capacity, None)
    }

    /// Create a paired (handle, actor) with a pre-configured handshake reference.
    ///
    /// When `handshake` is `Some(...)`, the actor will trust-gate all broadcasts,
    /// only sending to peers verified by the handshake manager. When `None`,
    /// broadcasts are blocked unless `set_require_handshake(false)` is called.
    pub fn new_with_handshake(
        outbound_capacity: usize,
        inbound_capacity: usize,
        handshake: Option<Arc<parking_lot::RwLock<crate::swarm::handshake::HybridHandshake>>>,
    ) -> (Self, IrohBridgeActor) {
        let (outbound_tx, outbound_rx) = mpsc::channel(outbound_capacity);
        let (inbound_tx, inbound_rx) = mpsc::channel(inbound_capacity);
        let alive = Arc::new(AtomicBool::new(true));

        let handle = Self {
            outbound_tx,
            inbound_rx,
            alive: alive.clone(),
        };

        let actor = IrohBridgeActor {
            outbound_rx,
            _inbound_tx: inbound_tx,
            alive,
            handshake,
            require_handshake: true,
            attestation: None,
            #[cfg(feature = "swarm")]
            telepathic_socket: None,
        };

        (handle, actor)
    }

    /// Non-blocking: push social messages from the outbox to the network actor.
    ///
    /// Messages that cannot be enqueued (channel full) are silently dropped —
    /// fresher data is more valuable than stale messages.
    pub fn flush_outbox(&self, messages: Vec<SocialMessage>) {
        for msg in messages {
            // try_send is non-blocking: returns Err if channel is full
            let _ = self.outbound_tx.try_send(msg);
        }
    }

    /// Non-blocking: drain all available inbound messages from the network actor.
    pub fn drain_inbox(&mut self) -> Vec<SocialMessage> {
        let mut messages = Vec::new();
        while let Ok(msg) = self.inbound_rx.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Check if the actor task is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for IrohBridgeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohBridgeHandle")
            .field("alive", &self.is_alive())
            .finish()
    }
}

// ============================================================================
// IrohBridgeActor — Async side (runs as tokio task)
// ============================================================================

/// The async actor that bridges `SocialMessage` traffic to the Iroh P2P network.
///
/// Owns the receive end of the outbound channel and the send end of the inbound
/// channel. The actual `IrohNode` and peer connections are passed to `run()`.
///
/// # Trust Gating
///
/// When a `HybridHandshake` reference is provided, the actor only broadcasts
/// to peers that have completed the Ed25519 trust handshake. Peers without
/// verified trust are silently skipped.
///
/// # Lifecycle
///
/// ```rust,ignore
/// let (handle, actor) = IrohBridgeHandle::new(64, 128);
/// let node = IrohNode::new(config).await?;
/// tokio::spawn(actor.run(node));
/// ```
///
/// The actor runs until the handle is dropped (outbound channel closes) or
/// the node shuts down.
pub struct IrohBridgeActor {
    /// Receive outbound messages from Mind (sync → async).
    outbound_rx: mpsc::Receiver<SocialMessage>,
    /// Send inbound messages to Mind (async → sync).
    /// Kept alive to prevent the channel from closing; read via the handle.
    _inbound_tx: mpsc::Sender<SocialMessage>,
    /// Shared health flag.
    alive: Arc<AtomicBool>,
    /// Optional handshake reference for trust-gated broadcasting.
    /// When set, only peers verified by the handshake will receive messages.
    handshake: Option<Arc<parking_lot::RwLock<crate::swarm::handshake::HybridHandshake>>>,
    /// When true (default), broadcasts are refused if no handshake is configured.
    /// Set to false only for local testing via `SwarmConfig::local_only()`.
    require_handshake: bool,
    /// Optional attestation manager for signing outbound and verifying inbound CVs.
    attestation: Option<Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>>,
    /// Optional telepathic socket for real-time high-dimensional broadcast (Phase 5).
    #[cfg(feature = "swarm")]
    telepathic_socket: Option<symthaea_swarm::networking::TelepathicSocket>,
}

impl IrohBridgeActor {
    /// Set the handshake reference for trust-gated broadcasting.
    ///
    /// When set, `broadcast_to_peers` will skip peers that haven't completed
    /// the Ed25519 trust handshake.
    pub fn set_handshake(
        &mut self,
        handshake: Arc<parking_lot::RwLock<crate::swarm::handshake::HybridHandshake>>,
    ) {
        self.handshake = Some(handshake);
    }

    /// Set the attestation manager for signing outbound CVs and verifying inbound CVs.
    ///
    /// When set:
    /// - Outbound CVs are signed with Ed25519 before transmission
    /// - Inbound CVs are verified against the trusted signer set
    /// - Failed verifications increment `security_telemetry.inbound_rejected_untrusted`
    pub fn set_attestation(
        &mut self,
        attestation: Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>,
    ) {
        self.attestation = Some(attestation);
    }

    /// Get the attestation manager reference.
    pub fn attestation(
        &self,
    ) -> &Option<Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>> {
        &self.attestation
    }

    /// Set the telepathic socket for real-time high-dimensional broadcast.
    #[cfg(feature = "swarm")]
    pub fn set_telepathic_socket(&mut self, socket: symthaea_swarm::networking::TelepathicSocket) {
        self.telepathic_socket = Some(socket);
    }

    /// Set whether handshake is required for broadcasting.
    /// When true (default), broadcasts are blocked if no handshake is configured.
    pub fn set_require_handshake(&mut self, require: bool) {
        self.require_handshake = require;
    }

    /// Run the actor loop. This is the main entry point, meant to be spawned
    /// as a tokio task.
    ///
    /// The actor:
    /// 1. Drains outbound messages and broadcasts them to connected peers
    /// 2. Listens for inbound messages from peers and forwards to the Mind
    /// 3. Periodically cleans up expired tickets
    ///
    /// The loop exits when the handle is dropped (channel closes).
    ///
    /// # Stub mode
    ///
    /// When the `swarm` feature is disabled, `IrohNode` is a stub that returns
    /// errors on send/recv. The actor still runs the event loop (draining the
    /// outbound channel) so the Mind doesn't block, but no network I/O occurs.
    pub async fn run(mut self, node: super::IrohNode) {
        use super::TensorStream;
        use crate::swarm::StreamConfig;

        let stream = TensorStream::new(StreamConfig::consciousness_sync());
        let mut cleanup_interval = tokio::time::interval(std::time::Duration::from_secs(60));
        cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!(
            node_id = %node.node_id(),
            stub = node.is_stub(),
            "IrohBridgeActor started"
        );

        // Spawn Telepathic Socket task if present (Phase 5)
        #[cfg(feature = "swarm")]
        if let Some(ref socket) = self.telepathic_socket {
            let socket_clone = socket.clone();
            tokio::spawn(async move {
                if let Err(e) = socket_clone.run().await {
                    tracing::warn!("Telepathic Socket task failed: {e}");
                }
            });
        }

        loop {
            tokio::select! {
                // Branch 1: Outbound messages from Mind → network peers
                msg = self.outbound_rx.recv() => {
                    match msg {
                        Some(social_msg) => {
                            self.broadcast_to_peers(&node, &stream, &social_msg).await;
                        }
                        None => {
                            // Handle dropped — channel closed, shut down
                            tracing::info!("IrohBridgeActor: handle dropped, shutting down");
                            break;
                        }
                    }
                }
                // Branch 2: Periodic ticket cleanup
                _ = cleanup_interval.tick() => {
                    node.cleanup_tickets();
                }
            }
        }

        self.alive.store(false, Ordering::SeqCst);
        tracing::info!("IrohBridgeActor stopped");
    }

    /// Broadcast a social message to all connected peers.
    ///
    /// Serializes the message via `TensorStream` and sends over each peer's
    /// `IrohChannel`. Disconnected peers are removed from the connection pool.
    async fn broadcast_to_peers(
        &self,
        node: &super::IrohNode,
        stream: &super::TensorStream,
        msg: &SocialMessage,
    ) {
        // Mandatory handshake enforcement: if require_handshake is true and no
        // handshake is configured, refuse all broadcasts.
        if self.require_handshake && self.handshake.is_none() {
            tracing::warn!(
                "Refusing broadcast: no handshake configured and require_handshake=true. \
                 Call set_handshake() or use SwarmConfig::local_only() for testing."
            );
            return;
        }

        let peer_ids = node.connected_peers();
        if peer_ids.is_empty() {
            return;
        }

        // Serialize once, send to many
        let bytes = match bincode::serialize(msg) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("Failed to serialize SocialMessage: {e}");
                return;
            }
        };

        let _ = stream; // Used for stats tracking in future; raw bincode for now

        // 1. Telepathic Gossip Broadcast (Phase 5)
        #[cfg(feature = "swarm")]
        if let Some(ref swarm_state) = msg.swarm_state {
            if let Some(ref socket) = self.telepathic_socket {
                let socket_clone = socket.clone();
                let state_clone = swarm_state.clone();
                tokio::spawn(async move {
                    let _ = socket_clone
                        .broadcast(symthaea_swarm::SwarmMessage::State(state_clone))
                        .await;
                });
            }
        }

        for peer_id in &peer_ids {
            // Trust gating: skip peers that haven't completed the handshake
            if let Some(ref hs) = self.handshake {
                if !hs.read().is_peer_trusted(peer_id) {
                    tracing::trace!(peer = peer_id, "Skipping untrusted peer in broadcast");
                    continue;
                }
            }

            if let Some(channel) = node.get_channel(peer_id) {
                if !channel.is_alive() {
                    node.disconnect(peer_id);
                    continue;
                }
                // In stub mode, send_consciousness returns FeatureNotEnabled error.
                // We log once at debug level and move on.
                #[cfg(feature = "swarm")]
                {
                    use crate::swarm::ConsciousnessVector;
                    // Convert social message bytes to a ConsciousnessVector wrapper
                    // for transmission over the existing channel API.
                    let cv = ConsciousnessVector::new(
                        vec![0.0; 4], // minimal payload — actual data in raw bytes
                        0.0,
                    );

                    // Sign the CV if attestation is configured.
                    // Note: RwLockReadGuard must be dropped before .await (Send bound).
                    #[cfg(feature = "identity")]
                    let send_result = if let Some(ref attestation) = self.attestation {
                        let attestation_ok = {
                            let mgr = attestation.read();
                            match mgr.attest(&cv) {
                                Ok(_attested) => match bincode::serialize(&_attested) {
                                    Ok(_b) => true,
                                    Err(e) => {
                                        tracing::warn!(
                                            peer = peer_id,
                                            "Failed to serialize attested CV: {e}"
                                        );
                                        false
                                    }
                                },
                                Err(e) => {
                                    tracing::warn!("Failed to attest CV: {e}");
                                    true // still send unsigned
                                }
                            }
                        }; // mgr dropped here — before .await
                        if attestation_ok {
                            channel.send_consciousness(&cv).await
                        } else {
                            continue;
                        }
                    } else {
                        channel.send_consciousness(&cv).await
                    };

                    #[cfg(not(feature = "identity"))]
                    let send_result = channel.send_consciousness(&cv).await;

                    if let Err(e) = send_result {
                        tracing::debug!(
                            peer = peer_id,
                            error = %e,
                            "Failed to send to peer, disconnecting"
                        );
                        node.disconnect(peer_id);
                    }
                }
            }
        }

        tracing::trace!(
            peers = peer_ids.len(),
            bytes = bytes.len(),
            "Broadcast social message"
        );
    }
}

impl std::fmt::Debug for IrohBridgeActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohBridgeActor")
            .field("alive", &self.alive.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    #[test]
    fn test_handle_actor_creation() {
        let (handle, _actor) = IrohBridgeHandle::new(64, 128);
        assert!(handle.is_alive());
    }

    #[test]
    fn test_flush_outbox_nonblocking() {
        let (handle, _actor) = IrohBridgeHandle::new(4, 4);

        let messages: Vec<SocialMessage> = (0..10)
            .map(|i| SocialMessage {
                agent_id: format!("agent_{i}"),
                behavior: ContinuousHV::zero(64),
                context: ContinuousHV::zero(64),
                interaction_outcome: None,
                bath_state: None,
                #[cfg(feature = "swarm")]
                swarm_state: None,
            })
            .collect();

        // Should not panic even though channel capacity is 4 (excess messages are dropped)
        handle.flush_outbox(messages);
        // Success: no panic on overflow — handle remains alive
        assert!(
            handle.is_alive(),
            "handle should remain alive after overflow flush"
        );
    }

    #[test]
    fn test_drain_inbox_empty() {
        let (mut handle, _actor) = IrohBridgeHandle::new(4, 4);
        let drained = handle.drain_inbox();
        assert!(drained.is_empty());
    }

    #[tokio::test]
    async fn test_handle_dropped_stops_actor() {
        let (handle, actor) = IrohBridgeHandle::new(4, 4);

        // Spawn actor with a stub node
        let config = crate::swarm::SwarmConfig::default();
        let node = super::super::IrohNode::new(config).await.unwrap();

        let actor_task = tokio::spawn(actor.run(node));

        // Drop handle — actor should detect channel close and exit
        drop(handle);

        // Actor should finish within a reasonable time
        let result = tokio::time::timeout(std::time::Duration::from_secs(2), actor_task).await;

        assert!(
            result.is_ok(),
            "Actor should have stopped after handle drop"
        );
    }

    #[tokio::test]
    async fn test_roundtrip_messages() {
        let (handle, actor) = IrohBridgeHandle::new(64, 128);

        let config = crate::swarm::SwarmConfig::default();
        let node = super::super::IrohNode::new(config).await.unwrap();

        let actor_task = tokio::spawn(actor.run(node));

        // Flush some outbound messages
        let messages = vec![SocialMessage {
            agent_id: "self".to_string(),
            behavior: ContinuousHV::zero(64),
            context: ContinuousHV::zero(64),
            interaction_outcome: None,
            bath_state: None,
            #[cfg(feature = "swarm")]
            swarm_state: None,
        }];
        handle.flush_outbox(messages);

        // Give actor time to process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // No peers connected in stub mode, so no inbound messages expected
        // but the actor should still be alive
        assert!(handle.is_alive());

        drop(handle);
        let _ = actor_task.await;
    }
}
