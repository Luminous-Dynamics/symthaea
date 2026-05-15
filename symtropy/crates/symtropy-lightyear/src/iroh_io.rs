// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! IrohIo — Bridge between Iroh QUIC transport and Lightyear's Link buffers.
//!
//! This is the "secret sauce" (~100 LOC) that gives Symtropy direct P2P
//! multiplayer with <50ms latency while getting all of Lightyear's netcode
//! features (prediction, rollback, interpolation) for free.
//!
//! # How It Works
//!
//! Lightyear's architecture is transport-agnostic. Each connection is a Bevy
//! entity with a `Link` component containing send/recv `VecDeque<Bytes>` buffers.
//! IO components (like this one) move bytes between the Link and the actual
//! transport in PreUpdate/PostUpdate.
//!
//! ```text
//! Iroh QUIC  ←→  IrohIo component  ←→  Link buffers  ←→  Lightyear netcode
//! ```

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bytes::Bytes;
use std::collections::VecDeque;

use symtropy_net::iroh_transport::IrohTransport;
use symtropy_net::transport::{Channel, Transport, TransportEvent};
use symtropy_net::PeerId;

/// IO component that bridges Iroh transport to Lightyear's Link buffers.
///
/// Attach this to a connection entity alongside Lightyear's `Link` component.
/// The `iroh_io_send` and `iroh_io_recv` systems handle the byte shuffling.
#[derive(Component)]
pub struct IrohIo {
    /// The underlying Iroh transport.
    pub transport: IrohTransport,
    /// Inbound buffer (filled from Iroh, drained by Lightyear).
    pub recv_buffer: VecDeque<Bytes>,
    /// Outbound buffer (filled by Lightyear, drained to Iroh).
    pub send_buffer: VecDeque<Bytes>,
}

impl IrohIo {
    /// Create a new IrohIo bridge for a peer.
    pub fn new(local_id: PeerId) -> Self {
        Self {
            transport: IrohTransport::new(local_id),
            recv_buffer: VecDeque::new(),
            send_buffer: VecDeque::new(),
        }
    }

    /// Connect to a room.
    pub fn connect(&mut self, room_id: &str) -> Result<(), String> {
        self.transport.connect(room_id)
    }
}

/// System: drain Iroh transport into recv_buffer (PreUpdate).
///
/// Called before Lightyear processes incoming packets.
pub fn iroh_io_recv(mut query: Query<&mut IrohIo>) {
    for mut io in query.iter_mut() {
        let events = io.transport.poll();
        for event in events {
            if let TransportEvent::Message(msg) = event {
                io.recv_buffer.push_back(Bytes::from(msg.data));
            }
        }
    }
}

/// System: flush send_buffer to Iroh transport (PostUpdate).
///
/// Called after Lightyear has produced outgoing packets.
pub fn iroh_io_send(mut query: Query<&mut IrohIo>) {
    for mut io in query.iter_mut() {
        while let Some(data) = io.send_buffer.pop_front() {
            // Broadcast to all connected peers on the reliable channel.
            // Lightyear handles its own reliability/ordering within the packet.
            let _ = io.transport.broadcast(Channel::Reliable, &data);
        }
    }
}

/// Plugin that adds IrohIo systems to the Bevy schedule.
pub struct IrohIoPlugin;

impl Plugin for IrohIoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, iroh_io_recv)
            .add_systems(PostUpdate, iroh_io_send);
    }
}
