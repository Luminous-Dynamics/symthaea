// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport trait — the abstraction over how peers exchange data.
//!
//! Symtropy uses two channels per peer connection:
//! - **Unreliable**: Physics state (positions, velocities). Dropped packets
//!   are fine — next tick's data supersedes. Low latency critical.
//! - **Reliable**: Authority changes, governance actions, chat.
//!   Must arrive in order, no drops.
//!
//! Implementations:
//! - `WebRtcTransport` (native, via webrtc-rs) — the production transport
//! - `LoopbackTransport` (testing) — in-memory, zero-latency
//! - Future: `web-sys` WebRTC for WASM browser builds

use crate::peer::PeerId;
use serde::{Deserialize, Serialize};

/// Channel reliability mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Unreliable, unordered. For physics state (positions, velocities).
    /// Dropped packets accepted — next tick supersedes.
    Unreliable,
    /// Reliable, ordered. For authority changes, governance, chat.
    Reliable,
}

/// A message received from a peer.
#[derive(Debug, Clone)]
pub struct PeerMessage {
    /// Which peer sent this.
    pub from: PeerId,
    /// Which channel it arrived on.
    pub channel: Channel,
    /// Raw payload bytes (MessagePack-encoded).
    pub data: Vec<u8>,
}

/// Events emitted by the transport layer.
#[derive(Debug, Clone)]
pub enum TransportEvent {
    /// A new peer connected.
    PeerConnected(PeerId),
    /// A peer disconnected.
    PeerDisconnected(PeerId),
    /// A message was received.
    Message(PeerMessage),
    /// Connection to signaling server established.
    SignalingConnected,
    /// Connection to signaling server lost.
    SignalingDisconnected,
    /// Error occurred.
    Error(String),
}

/// The transport trait — how peers actually exchange bytes.
///
/// Implementations handle signaling, ICE negotiation, and data channels.
/// The game loop calls `poll()` each tick and `send()` when it has
/// state to distribute.
pub trait Transport {
    /// Connect to the signaling server and join a room.
    /// This begins the peer discovery process.
    fn connect(&mut self, room_id: &str) -> Result<(), String>;

    /// Disconnect from all peers and the signaling server.
    fn disconnect(&mut self);

    /// Send data to a specific peer on the given channel.
    fn send(&mut self, to: PeerId, channel: Channel, data: &[u8]) -> Result<(), String>;

    /// Broadcast data to all connected peers on the given channel.
    fn broadcast(&mut self, channel: Channel, data: &[u8]) -> Result<(), String>;

    /// Poll for events. Call once per game tick.
    /// Returns all events that occurred since the last poll.
    fn poll(&mut self) -> Vec<TransportEvent>;

    /// Number of currently connected peers.
    fn peer_count(&self) -> usize;

    /// Whether we're connected to the signaling server.
    fn is_signaling_connected(&self) -> bool;

    /// Our own peer ID (assigned by signaling server or self-generated).
    fn local_peer_id(&self) -> PeerId;
}

/// Physics state snapshot for a single body — sent on the unreliable channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyStateUpdate {
    /// Body handle identifier.
    pub body_id: u32,
    /// Position (x, y, z) — or (x, y) for 2D.
    pub position: [f64; 3],
    /// Linear velocity.
    pub velocity: [f64; 3],
    /// Rotation (quaternion: w, x, y, z).
    pub rotation: [f64; 4],
    /// Angular velocity.
    pub angular_velocity: [f64; 3],
}

/// A batch of body state updates — one per tick from the authority peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsSync {
    /// Tick number this snapshot is from.
    pub tick: u64,
    /// The peer that computed this physics step.
    pub authority: u64,
    /// Body states (only bodies this peer has authority over).
    pub bodies: Vec<BodyStateUpdate>,
}

/// Authority change message — sent on the reliable channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorityMessage {
    /// Claim authority over bodies.
    Claim { body_ids: Vec<u32>, peer_id: u64 },
    /// Release authority over bodies.
    Release { body_ids: Vec<u32> },
    /// Request authority transfer (negotiation).
    RequestTransfer { body_ids: Vec<u32>, reason: String },
}
