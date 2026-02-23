//! Mesh Bridge — Async/Sync Actor Pattern for the Mycelial Mesh.
//!
//! Connects the synchronous `ContinuousMind` tick loop to the async
//! `DualLayerMesh` networking layer via bounded mpsc channels.
//!
//! ```text
//! ┌──────────────────────┐       mpsc        ┌──────────────────────┐
//! │  ContinuousMind      │  ──────────────►  │  MeshBridgeActor     │
//! │  (sync, 50Hz)        │  mesh_outbox      │  (async, tokio task)  │
//! │                      │  ◄──────────────  │                      │
//! │  MeshBridgeHandle    │  mesh_inbox       │  DualLayerMesh       │
//! └──────────────────────┘                   └──────────────────────┘
//! ```
//!
//! Mirrors the `IrohBridgeHandle` / `IrohBridgeActor` pattern — see
//! `swarm/iroh/bridge.rs` for the original design rationale.

use super::{DualLayerMesh, MeshReceiver, WisdomPacket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

// ============================================================================
// MeshOutbound — envelope for outgoing mesh packets
// ============================================================================

/// An outgoing mesh packet queued for transmission.
#[derive(Clone)]
pub struct MeshOutbound {
    /// The wisdom packet to transmit over the mesh.
    pub packet: WisdomPacket,
}

impl std::fmt::Debug for MeshOutbound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshOutbound")
            .field("packet", &self.packet)
            .finish()
    }
}

// ============================================================================
// MeshBridgeHandle — Sync side (held by ContinuousMind)
// ============================================================================

/// A sync-safe handle for the Mind to communicate with the async mesh actor.
///
/// All methods are non-blocking (`try_send` / `try_recv`), so they are safe
/// to call from the 50Hz synchronous tick loop without risking deadlocks.
pub struct MeshBridgeHandle {
    /// Send outbound mesh packets to the actor (non-blocking try_send).
    outbound_tx: mpsc::Sender<MeshOutbound>,
    /// Receive inbound wisdom packets from the actor (non-blocking try_recv).
    inbound_rx: mpsc::Receiver<WisdomPacket>,
    /// Health flag — `false` if the actor task has exited.
    alive: Arc<AtomicBool>,
}

impl MeshBridgeHandle {
    /// Create a paired (handle, actor) with the given channel capacities.
    ///
    /// - `outbound_capacity`: Bounded channel from Mind → Actor
    /// - `inbound_capacity`: Bounded channel from Actor → Mind
    pub fn new(outbound_capacity: usize, inbound_capacity: usize) -> (Self, MeshBridgeActor) {
        let (outbound_tx, outbound_rx) = mpsc::channel(outbound_capacity);
        let (inbound_tx, inbound_rx) = mpsc::channel(inbound_capacity);
        let alive = Arc::new(AtomicBool::new(true));

        let handle = Self {
            outbound_tx,
            inbound_rx,
            alive: alive.clone(),
        };

        let actor = MeshBridgeActor {
            outbound_rx,
            inbound_tx,
            alive,
        };

        (handle, actor)
    }

    /// Non-blocking: push mesh outbound packets to the network actor.
    ///
    /// Packets that cannot be enqueued (channel full) are silently dropped —
    /// fresher wisdom is more valuable than stale packets.
    pub fn flush_outbox(&self, packets: Vec<MeshOutbound>) {
        for pkt in packets {
            let _ = self.outbound_tx.try_send(pkt);
        }
    }

    /// Non-blocking: drain all available inbound wisdom packets from the actor.
    pub fn drain_inbox(&mut self) -> Vec<WisdomPacket> {
        let mut packets = Vec::new();
        while let Ok(pkt) = self.inbound_rx.try_recv() {
            packets.push(pkt);
        }
        packets
    }

    /// Check if the actor task is still alive.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for MeshBridgeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshBridgeHandle")
            .field("alive", &self.is_alive())
            .finish()
    }
}

// ============================================================================
// MeshBridgeActor — Async side (runs as tokio task)
// ============================================================================

/// The async actor that bridges mesh traffic between the Mind and the
/// physical `DualLayerMesh` radio network.
///
/// Owns the receive end of the outbound channel and the send end of the
/// inbound channel. The actual `DualLayerMesh` is passed to `run()`.
///
/// # Lifecycle
///
/// ```rust,ignore
/// let (handle, actor) = MeshBridgeHandle::new(64, 128);
/// let mesh = DualLayerMesh::new(node_id);
/// tokio::spawn(actor.run(mesh));
/// ```
///
/// The actor runs until the handle is dropped (outbound channel closes).
pub struct MeshBridgeActor {
    /// Receive outbound packets from Mind (sync → async).
    outbound_rx: mpsc::Receiver<MeshOutbound>,
    /// Send inbound wisdom packets to Mind (async → sync).
    inbound_tx: mpsc::Sender<WisdomPacket>,
    /// Shared health flag.
    alive: Arc<AtomicBool>,
}

impl MeshBridgeActor {
    /// Run the actor loop. Meant to be spawned as a tokio task.
    ///
    /// The actor:
    /// 1. Receives `MeshOutbound` from Mind, sends via `DualLayerMesh`
    /// 2. Polls DualLayerMesh transports for incoming data, feeds `MeshReceiver`
    /// 3. Completed WisdomPackets pushed to inbound channel
    ///
    /// The loop exits when the handle is dropped (channel closes).
    pub async fn run(mut self, mesh: DualLayerMesh, mut receiver: MeshReceiver) {
        let mut poll_interval = tokio::time::interval(std::time::Duration::from_millis(50));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!("MeshBridgeActor started");

        loop {
            tokio::select! {
                // Branch 1: Outbound packets from Mind → mesh radio
                msg = self.outbound_rx.recv() => {
                    match msg {
                        Some(outbound) => {
                            match mesh.send(&outbound.packet) {
                                Ok(route) => {
                                    tracing::trace!(
                                        route = %route,
                                        urgency = ?outbound.packet.urgency,
                                        "Sent wisdom packet over mesh"
                                    );
                                }
                                Err(e) => {
                                    tracing::debug!(
                                        error = %e,
                                        urgency = ?outbound.packet.urgency,
                                        "Failed to send packet over mesh"
                                    );
                                }
                            }
                        }
                        None => {
                            // Handle dropped — channel closed, shut down
                            tracing::info!("MeshBridgeActor: handle dropped, shutting down");
                            break;
                        }
                    }
                }
                // Branch 2: Poll for incoming radio data
                _ = poll_interval.tick() => {
                    // Expire stale fragment assemblies
                    receiver.expire_stale();

                    // Poll mesh transports for incoming data
                    for packet in mesh.poll_incoming(&mut receiver) {
                        if self.inbound_tx.try_send(packet).is_err() {
                            tracing::trace!("Inbound channel full, dropping wisdom packet");
                        }
                    }
                }
            }
        }

        self.alive.store(false, Ordering::SeqCst);
        tracing::info!("MeshBridgeActor stopped");
    }
}

impl std::fmt::Debug for MeshBridgeActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshBridgeActor")
            .field("alive", &self.alive.load(Ordering::Relaxed))
            .finish()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_actor_creation() {
        let (handle, _actor) = MeshBridgeHandle::new(64, 128);
        assert!(handle.is_alive());
    }

    #[test]
    fn flush_outbox_nonblocking() {
        let (handle, _actor) = MeshBridgeHandle::new(4, 4);

        let packets: Vec<MeshOutbound> = (0..10)
            .map(|i| MeshOutbound {
                packet: WisdomPacket {
                    source_id: [0; 8],
                    sequence: i,
                    phi: 0.5,
                    urgency: super::super::MeshUrgency::Normal,
                    timestamp_s: 0,
                    payload_type: super::super::PayloadType::WisdomVector,
                    auth_mac: 0,
                    ttl: 0,
                    wisdom: symthaea_core::hdc::BinaryHV([0u8; 2048]),
                },
            })
            .collect();

        // Should not panic even though channel capacity is 4
        handle.flush_outbox(packets);
    }

    #[test]
    fn drain_inbox_empty() {
        let (mut handle, _actor) = MeshBridgeHandle::new(4, 4);
        let drained = handle.drain_inbox();
        assert!(drained.is_empty());
    }

    #[test]
    fn alive_flag_shared() {
        let (handle, actor) = MeshBridgeHandle::new(4, 4);
        assert!(handle.is_alive());

        // Manually flip the flag to simulate actor shutdown
        actor.alive.store(false, Ordering::SeqCst);
        assert!(!handle.is_alive());
    }
}
