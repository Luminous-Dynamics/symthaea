// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Iroh transport — direct P2P via QUIC (Magicsock NAT traversal).
//!
//! Uses Iroh's QUIC-based networking for <50ms latency peer connections.
//! NAT traversal is handled automatically by Magicsock — no STUN/TURN
//! configuration needed in most cases.
//!
//! This is the **preferred transport for native (desktop/server) builds**.
//! For WASM/browser builds, use `RelayTransport` (WebSocket fallback).
//!
//! # Architecture
//!
//! ```text
//! IrohTransport
//!   └→ IrohBridgeHandle (from symthaea::swarm::iroh)
//!        └→ IrohBridgeActor (async tokio task)
//!             └→ Iroh Endpoint (QUIC + Magicsock)
//!                  ├→ Reliable streams (authority messages)
//!                  └→ Unreliable datagrams (physics state)
//! ```
//!
//! # Integration with Symthaea
//!
//! Symthaea's Iroh module (`symthaea/src/swarm/iroh/`) provides:
//! - `IrohBridgeHandle` — sync-safe handle for the game tick loop
//! - `IrohBridgeActor` — async actor managing connections
//! - `TensorStream` — bidirectional streaming (we reuse for physics)
//! - `HybridHandshake` — trust-gated peer verification
//!
//! Symtropy wraps this in the `Transport` trait so the game code
//! doesn't know or care whether it's using Iroh, WebRTC, or loopback.
//!
//! # Honest status (verified 2026-07-04): this is a stub, not a transport
//!
//! **Nothing in this file talks to the network.** There is no `iroh`
//! dependency in this crate's `Cargo.toml`, no QUIC endpoint, no
//! Magicsock, no NAT traversal, no discovery, and no bytes ever cross a
//! process boundary. `IrohTransport` is an in-process queue:
//! `send()`/`broadcast()` push into an `outbox: Vec<...>` and `poll()`
//! drains an `inbox: Vec<PeerMessage>` — both are ordinary in-memory
//! `Vec`s that nothing outside this struct ever reads from or writes to
//! (the `inject_*`/`drain_outbox` methods exist for a future bridge actor
//! to call, but no such actor exists yet). `connect()` just flips a bool;
//! it never joins any swarm. The 4 unit tests below pass because they
//! call `inject_peer_connected`/`inject_message` directly — they exercise
//! the queue plumbing, not any P2P connectivity.
//!
//! What *does* exist, and is real: a substantial (~2,143 LOC across
//! `bridge.rs`/`mod.rs`/`streaming.rs`/`ticket.rs`) Iroh integration in
//! the separate `symthaea` crate at `symthaea/src/swarm/iroh/`, built on
//! a real, patched `iroh = "0.97"` dependency
//! (`symthaea/Cargo.toml:379,1912`, patch at
//! `symthaea/patches/iroh/iroh/`). It provides `IrohBridgeHandle` /
//! `IrohBridgeActor` and does real QUIC endpoint setup, connection
//! lifecycle, and streaming — but it lives inside `symthaea`'s `swarm`
//! module, gated behind `symthaea`'s own feature flags, and is not
//! exposed as a small, reusable library API that `symtropy-net` could
//! cleanly depend on today.
//!
//! Wiring this transport for real is **not** a small bolt-on:
//! - Depending on the full `symthaea` crate just to reach `swarm::iroh`
//!   would be architecturally wrong (a physics/game-networking crate
//!   pulling in an entire consciousness engine).
//! - The alternative — depending directly on `iroh` and writing a fresh
//!   endpoint/connect/accept/discovery loop against Iroh's real API,
//!   independent of `symthaea::swarm::iroh` — is genuine, non-trivial
//!   async networking work requiring real understanding of Iroh's
//!   `Endpoint`, ALPN protocol negotiation, `NodeAddr`/discovery, and
//!   connection lifecycle, plus real multi-process testing.
//! - Either path is scoped as its own follow-up task, not something to
//!   rush alongside unrelated fixes. A half-wired transport that *looks*
//!   connected while silently dropping every packet would be strictly
//!   worse than this honestly-labeled stub.
//!
//! Until that follow-up lands, `LoopbackTransport` (single-process,
//! in-memory) is the only transport in this crate proven to move bytes
//! between peers, and (with the `webrtc` feature, fixed 2026-07-04)
//! `RelayTransport` compiles and can reach a real WebSocket signaling
//! server, though it hasn't been validated end-to-end against one.

use crate::peer::PeerId;
use crate::transport::{Channel, PeerMessage, Transport, TransportEvent};

/// Iroh-backed transport using QUIC for direct P2P connections.
///
/// **This is a stub — see the module-level doc comment above for the
/// full, honest breakdown of what is and isn't real here.** In short:
/// every method below only touches in-memory `Vec` queues on `self`;
/// nothing here talks to Iroh, QUIC, or the network. It will be wired to
/// a real Iroh transport in a dedicated follow-up.
///
/// The integration path (still true, still future work):
/// 1. Symtropy enables `symthaea/swarm` feature, or `symtropy-net`
///    depends on `iroh` directly and gets its own endpoint code.
/// 2. Creates `IrohBridgeHandle` + `IrohBridgeActor` at startup (or the
///    equivalent fresh endpoint/actor if built directly on `iroh`).
/// 3. Wraps the handle in `IrohTransport`.
/// 4. Passes to `NetworkSession<IrohTransport>`.
/// 5. Game loop calls `session.tick()` — physics flows over QUIC.
pub struct IrohTransport {
    local_id: PeerId,
    peers: Vec<PeerId>,
    connected: bool,
    /// Outbound message queue (drained by the bridge actor).
    outbox: Vec<(PeerId, Channel, Vec<u8>)>,
    /// Inbound message queue (filled by the bridge actor).
    inbox: Vec<PeerMessage>,
}

impl IrohTransport {
    /// Create a new Iroh transport.
    ///
    /// In the full integration, this takes an `IrohBridgeHandle` parameter.
    /// For now, it creates a standalone instance that queues messages
    /// until the bridge is wired.
    pub fn new(local_id: PeerId) -> Self {
        Self {
            local_id,
            peers: Vec::new(),
            connected: false,
            outbox: Vec::new(),
            inbox: Vec::new(),
        }
    }

    /// Inject a received message (called by the bridge actor adapter).
    pub fn inject_message(&mut self, msg: PeerMessage) {
        self.inbox.push(msg);
    }

    /// Inject a peer connection event.
    pub fn inject_peer_connected(&mut self, peer: PeerId) {
        if !self.peers.contains(&peer) {
            self.peers.push(peer);
        }
    }

    /// Inject a peer disconnection event.
    pub fn inject_peer_disconnected(&mut self, peer: PeerId) {
        self.peers.retain(|p| *p != peer);
    }

    /// Drain outbound messages (consumed by the bridge actor adapter).
    pub fn drain_outbox(&mut self) -> Vec<(PeerId, Channel, Vec<u8>)> {
        std::mem::take(&mut self.outbox)
    }
}

impl Transport for IrohTransport {
    fn connect(&mut self, _room_id: &str) -> Result<(), String> {
        // In the full integration, this joins the Iroh swarm.
        // The room_id maps to a topic/document hash.
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) {
        self.connected = false;
        self.peers.clear();
        self.outbox.clear();
        self.inbox.clear();
    }

    fn send(&mut self, to: PeerId, channel: Channel, data: &[u8]) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".into());
        }
        self.outbox.push((to, channel, data.to_vec()));
        Ok(())
    }

    fn broadcast(&mut self, channel: Channel, data: &[u8]) -> Result<(), String> {
        if !self.connected {
            return Err("Not connected".into());
        }
        for peer in &self.peers {
            self.outbox.push((*peer, channel, data.to_vec()));
        }
        Ok(())
    }

    fn poll(&mut self) -> Vec<TransportEvent> {
        let mut events = Vec::new();
        for msg in self.inbox.drain(..) {
            events.push(TransportEvent::Message(msg));
        }
        events
    }

    fn peer_count(&self) -> usize {
        self.peers.len()
    }

    fn is_signaling_connected(&self) -> bool {
        self.connected
    }

    fn local_peer_id(&self) -> PeerId {
        self.local_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iroh_transport_basic() {
        let mut transport = IrohTransport::new(PeerId(1));
        transport.connect("test-room").unwrap();
        assert!(transport.is_signaling_connected());
        assert_eq!(transport.peer_count(), 0);

        // Simulate peer joining
        transport.inject_peer_connected(PeerId(2));
        assert_eq!(transport.peer_count(), 1);

        // Send a message
        transport
            .send(PeerId(2), Channel::Unreliable, b"physics")
            .unwrap();
        let outbox = transport.drain_outbox();
        assert_eq!(outbox.len(), 1);
        assert_eq!(outbox[0].0, PeerId(2));
        assert_eq!(outbox[0].2, b"physics");
    }

    #[test]
    fn iroh_transport_receive() {
        let mut transport = IrohTransport::new(PeerId(1));
        transport.connect("test").unwrap();

        transport.inject_message(PeerMessage {
            from: PeerId(2),
            channel: Channel::Reliable,
            data: b"authority".to_vec(),
        });

        let events = transport.poll();
        assert_eq!(events.len(), 1);
        match &events[0] {
            TransportEvent::Message(msg) => {
                assert_eq!(msg.from, PeerId(2));
                assert_eq!(msg.data, b"authority");
            }
            _ => panic!("Expected Message"),
        }
    }

    #[test]
    fn iroh_transport_broadcast() {
        let mut transport = IrohTransport::new(PeerId(0));
        transport.connect("test").unwrap();
        transport.inject_peer_connected(PeerId(1));
        transport.inject_peer_connected(PeerId(2));

        transport.broadcast(Channel::Unreliable, b"state").unwrap();
        let outbox = transport.drain_outbox();
        assert_eq!(outbox.len(), 2);
    }

    #[test]
    fn iroh_transport_disconnect() {
        let mut transport = IrohTransport::new(PeerId(0));
        transport.connect("test").unwrap();
        transport.inject_peer_connected(PeerId(1));

        transport.disconnect();
        assert!(!transport.is_signaling_connected());
        assert_eq!(transport.peer_count(), 0);
    }
}
