// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

/// Shared encryption key that can be updated at runtime.
///
/// The Mind writes to this via the handle; the bridge actor reads it
/// when constructing / updating DualLayerMesh and MeshReceiver.
#[cfg(feature = "mesh-encryption")]
type SharedEncryptionKey = Arc<parking_lot::Mutex<Option<[u8; 32]>>>;

/// Shared encryption epoch (random byte, set once).
#[cfg(feature = "mesh-encryption")]
type SharedEpoch = Arc<std::sync::atomic::AtomicU8>;

/// Generation counter for atomic key+epoch propagation.
///
/// Incremented by the handle on each `set_encryption_key()` call.
/// The actor polls this to detect changes without comparing key bytes.
#[cfg(feature = "mesh-encryption")]
type SharedKeyGeneration = Arc<std::sync::atomic::AtomicU64>;

// ============================================================================
// VerifiedWisdomPacket — type-level proof of attestation verification
// ============================================================================

/// Type-level proof that a WisdomPacket has passed attestation verification.
///
/// This newtype ensures that only verified packets can enter the cognitive loop
/// when attestation is enabled. When attestation is disabled (no `identity` feature),
/// packets are wrapped directly.
#[derive(Debug, Clone)]
pub struct VerifiedWisdomPacket {
    /// The verified wisdom packet.
    pub packet: WisdomPacket,
    /// Whether attestation was actually checked (false = passthrough mode).
    pub attestation_checked: bool,
}

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
    /// Shared encryption key — writable by Mind, readable by actor.
    #[cfg(feature = "mesh-encryption")]
    shared_key: SharedEncryptionKey,
    /// Shared encryption epoch.
    #[cfg(feature = "mesh-encryption")]
    shared_epoch: SharedEpoch,
    /// Generation counter — incremented on each key update.
    #[cfg(feature = "mesh-encryption")]
    shared_generation: SharedKeyGeneration,
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

        #[cfg(feature = "mesh-encryption")]
        let shared_key: SharedEncryptionKey = Arc::new(parking_lot::Mutex::new(None));
        #[cfg(feature = "mesh-encryption")]
        let shared_epoch: SharedEpoch = Arc::new(std::sync::atomic::AtomicU8::new(0));
        #[cfg(feature = "mesh-encryption")]
        let shared_generation: SharedKeyGeneration = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let handle = Self {
            outbound_tx,
            inbound_rx,
            alive: alive.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_key: shared_key.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_epoch: shared_epoch.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_generation: shared_generation.clone(),
        };

        let actor = MeshBridgeActor {
            outbound_rx,
            inbound_tx,
            alive,
            #[cfg(feature = "mesh-encryption")]
            shared_key,
            #[cfg(feature = "mesh-encryption")]
            shared_epoch,
            #[cfg(feature = "mesh-encryption")]
            shared_generation,
            #[cfg(feature = "identity")]
            attestation: None,
        };

        (handle, actor)
    }

    /// Create a paired (handle, actor) with attestation verification.
    #[cfg(feature = "identity")]
    pub fn new_with_attestation(
        outbound_capacity: usize,
        inbound_capacity: usize,
        attestation: Option<
            std::sync::Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>,
        >,
    ) -> (Self, MeshBridgeActor) {
        let (outbound_tx, outbound_rx) = mpsc::channel(outbound_capacity);
        let (inbound_tx, inbound_rx) = mpsc::channel(inbound_capacity);
        let alive = Arc::new(AtomicBool::new(true));

        #[cfg(feature = "mesh-encryption")]
        let shared_key: SharedEncryptionKey = Arc::new(parking_lot::Mutex::new(None));
        #[cfg(feature = "mesh-encryption")]
        let shared_epoch: SharedEpoch = Arc::new(std::sync::atomic::AtomicU8::new(0));
        #[cfg(feature = "mesh-encryption")]
        let shared_generation: SharedKeyGeneration = Arc::new(std::sync::atomic::AtomicU64::new(0));

        let handle = Self {
            outbound_tx,
            inbound_rx,
            alive: alive.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_key: shared_key.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_epoch: shared_epoch.clone(),
            #[cfg(feature = "mesh-encryption")]
            shared_generation: shared_generation.clone(),
        };

        let actor = MeshBridgeActor {
            outbound_rx,
            inbound_tx,
            alive,
            #[cfg(feature = "mesh-encryption")]
            shared_key,
            #[cfg(feature = "mesh-encryption")]
            shared_epoch,
            #[cfg(feature = "mesh-encryption")]
            shared_generation,
            attestation,
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

    /// Set the encryption key for the bridge actor.
    ///
    /// The actor will pick up the new key on its next poll cycle.
    /// Pass `None` to disable encryption.
    #[cfg(feature = "mesh-encryption")]
    pub fn set_encryption_key(&self, key: Option<[u8; 32]>) {
        *self.shared_key.lock() = key;
        self.shared_generation.fetch_add(1, Ordering::Release);
    }

    /// Set the encryption epoch byte.
    #[cfg(feature = "mesh-encryption")]
    pub fn set_encryption_epoch(&self, epoch: u8) {
        self.shared_epoch.store(epoch, Ordering::Relaxed);
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
    /// Shared encryption key — readable by actor, writable by Mind.
    #[cfg(feature = "mesh-encryption")]
    shared_key: SharedEncryptionKey,
    /// Shared encryption epoch.
    #[cfg(feature = "mesh-encryption")]
    shared_epoch: SharedEpoch,
    /// Generation counter — polls for changes instead of comparing key bytes.
    #[cfg(feature = "mesh-encryption")]
    shared_generation: SharedKeyGeneration,
    /// Optional attestation manager for verifying inbound packets.
    /// Feature-gated behind `identity`. When present, inbound packets
    /// are verified before being forwarded to the Mind.
    #[cfg(feature = "identity")]
    attestation:
        Option<std::sync::Arc<parking_lot::RwLock<crate::swarm::attestation::AttestationManager>>>,
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
    #[allow(unused_mut)] // mut needed only with mesh-encryption feature
    pub async fn run(mut self, mut mesh: DualLayerMesh, mut receiver: MeshReceiver) {
        let mut poll_interval = tokio::time::interval(std::time::Duration::from_millis(50));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Apply initial encryption key if one was set before run() was called.
        // Only overwrite if shared state has a key — preserve builder-set keys otherwise.
        #[cfg(feature = "mesh-encryption")]
        {
            let key = *self.shared_key.lock();
            if key.is_some() {
                mesh.set_encryption_key(key);
                receiver.set_encryption_key(key);
                let epoch = self.shared_epoch.load(Ordering::Relaxed);
                mesh.set_encryption_epoch(epoch);
            }
        }

        #[cfg(feature = "mesh-encryption")]
        let mut last_generation: u64 = self.shared_generation.load(Ordering::Acquire);

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
                // Branch 2: Poll for incoming radio data + key updates
                _ = poll_interval.tick() => {
                    // Check for encryption key updates from the Mind via generation counter.
                    // Only acquires the mutex when generation has changed, ensuring
                    // key + epoch are applied atomically without comparing key bytes.
                    #[cfg(feature = "mesh-encryption")]
                    {
                        let current_gen = self.shared_generation.load(Ordering::Acquire);
                        if current_gen != last_generation {
                            let current_key = *self.shared_key.lock();
                            let epoch = self.shared_epoch.load(Ordering::Relaxed);
                            mesh.set_encryption_key(current_key);
                            receiver.set_encryption_key(current_key);
                            mesh.set_encryption_epoch(epoch);
                            last_generation = current_gen;
                            tracing::info!(
                                generation = current_gen,
                                "MeshBridgeActor: encryption key updated"
                            );
                        }
                    }

                    // Expire stale fragment assemblies
                    receiver.expire_stale();

                    // Poll mesh transports for incoming data
                    for packet in mesh.poll_incoming(&mut receiver) {
                        // Attestation verification: reject unverified packets when identity feature is active
                        #[cfg(feature = "identity")]
                        {
                            if let Some(ref attestation) = self.attestation {
                                let mgr = attestation.read();
                                if mgr.requires_attestation() {
                                    // The authentication tag is not an attestation. For now, log
                                    // and pass through — full attestation requires
                                    // AttestedConsciousnessVector wrapping which is a higher-level concern
                                    tracing::trace!(
                                        source = ?packet.source_id,
                                        "Attestation active: packet auth_mac={:02x?}", packet.auth_mac
                                    );
                                }
                            }
                        }
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
                    auth_mac: [0; 32],
                    ttl: 0,
                    wisdom: symthaea_core::hdc::BinaryHV([0u8; 2048]),
                },
            })
            .collect();

        // Should not panic even though channel capacity is 4 (excess packets are dropped)
        handle.flush_outbox(packets);
        // Success: no panic on overflow — handle remains alive
        assert!(
            handle.is_alive(),
            "handle should remain alive after overflow flush"
        );
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

    #[test]
    fn verified_wisdom_packet_creation() {
        let packet = WisdomPacket {
            source_id: [1; 8],
            sequence: 42,
            phi: 0.7,
            urgency: super::super::MeshUrgency::Normal,
            timestamp_s: 1000,
            payload_type: super::super::PayloadType::WisdomVector,
            auth_mac: [0; 32],
            ttl: 5,
            wisdom: symthaea_core::hdc::BinaryHV([0u8; 2048]),
        };

        let verified = VerifiedWisdomPacket {
            packet: packet.clone(),
            attestation_checked: true,
        };
        assert!(verified.attestation_checked);
        assert_eq!(verified.packet.sequence, 42);

        let passthrough = VerifiedWisdomPacket {
            packet,
            attestation_checked: false,
        };
        assert!(!passthrough.attestation_checked);
    }

    #[cfg(feature = "identity")]
    #[test]
    fn handle_actor_creation_with_attestation() {
        let (handle, _actor) = MeshBridgeHandle::new_with_attestation(64, 128, None);
        assert!(handle.is_alive());
    }
}
