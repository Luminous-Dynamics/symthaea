// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network session — ties transport, authority, and state sync together.
//!
//! The game loop calls `session.tick()` each frame, which:
//! 1. Polls the transport for incoming messages
//! 2. Applies received physics state to remote bodies
//! 3. Sends local authority bodies' state to peers
//! 4. Handles authority claim/release messages

use std::collections::HashMap;

use crate::authority::SpatialAuthority;
use crate::config::NetworkConfig;
use crate::peer::{PeerId, PeerState};
use crate::transport::{
    AuthorityMessage, BodyStateUpdate, Channel, PhysicsSync, Transport, TransportEvent,
};

/// A multiplayer network session.
pub struct NetworkSession<T: Transport> {
    /// The underlying transport (WebRTC, loopback, etc.).
    pub transport: T,
    /// Network configuration.
    pub config: NetworkConfig,
    /// Spatial authority system.
    pub authority: SpatialAuthority,
    /// Known peers.
    pub peers: HashMap<PeerId, PeerState>,
    /// Current tick.
    pub tick: u64,
    /// Received physics updates from remote peers (consumed by game loop).
    pub incoming_physics: Vec<PhysicsSync>,
    /// Received authority messages (consumed by game loop).
    pub incoming_authority: Vec<AuthorityMessage>,
}

impl<T: Transport> NetworkSession<T> {
    /// Create a new session with the given transport and config.
    pub fn new(transport: T, config: NetworkConfig) -> Self {
        let local_id = transport.local_peer_id();
        Self {
            transport,
            config,
            authority: SpatialAuthority::new(local_id, 100.0),
            peers: HashMap::new(),
            tick: 0,
            incoming_physics: Vec::new(),
            incoming_authority: Vec::new(),
        }
    }

    /// Join a room and start peer discovery.
    pub fn join(&mut self, room_id: &str) -> Result<(), String> {
        self.transport.connect(room_id)
    }

    /// Leave the session.
    pub fn leave(&mut self) {
        self.transport.disconnect();
        self.peers.clear();
    }

    /// Process one tick of networking.
    ///
    /// Call this once per game tick. It:
    /// 1. Polls transport for events
    /// 2. Handles peer connect/disconnect
    /// 3. Deserializes incoming physics and authority messages
    /// 4. Clears stale data from previous tick
    pub fn tick(&mut self) {
        self.tick += 1;
        self.incoming_physics.clear();
        self.incoming_authority.clear();

        let events = self.transport.poll();
        for event in events {
            match event {
                TransportEvent::PeerConnected(peer_id) => {
                    let peer = PeerState::remote(peer_id, format!("Peer-{}", peer_id.0), 0);
                    self.peers.insert(peer_id, peer);
                }
                TransportEvent::PeerDisconnected(peer_id) => {
                    self.peers.remove(&peer_id);
                }
                TransportEvent::Message(msg) => {
                    // Update last-seen
                    if let Some(peer) = self.peers.get_mut(&msg.from) {
                        peer.mark_seen(self.tick);
                    }

                    match msg.channel {
                        Channel::Unreliable => {
                            // Physics state update
                            if let Ok(sync) = rmp_serde::from_slice::<PhysicsSync>(&msg.data) {
                                self.incoming_physics.push(sync);
                            }
                        }
                        Channel::Reliable => {
                            // Authority message
                            if let Ok(auth) = rmp_serde::from_slice::<AuthorityMessage>(&msg.data) {
                                self.incoming_authority.push(auth);
                            }
                        }
                    }
                }
                TransportEvent::SignalingConnected => {}
                TransportEvent::SignalingDisconnected => {}
                TransportEvent::Error(e) => {
                    // Log but don't crash — graceful degradation
                    #[cfg(feature = "logging")]
                    eprintln!("[symtropy-net] Transport error: {e}");
                    let _ = e;
                }
            }
        }

        // Expire stale peers
        let timeout =
            self.config.authority_timeout_ms / (1000 / self.config.send_rate_hz as u64).max(1);
        let stale: Vec<PeerId> = self
            .peers
            .iter()
            .filter(|(_, p)| !p.is_alive(self.tick, timeout))
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.peers.remove(&id);
        }
    }

    /// Send local physics state to all peers.
    ///
    /// Call after computing physics for bodies the local peer has authority over.
    pub fn send_physics(&mut self, bodies: Vec<BodyStateUpdate>) {
        if bodies.is_empty() || self.transport.peer_count() == 0 {
            return;
        }

        let sync = PhysicsSync {
            tick: self.tick,
            authority: self.transport.local_peer_id().0,
            bodies,
        };

        if let Ok(data) = rmp_serde::to_vec(&sync) {
            let _ = self.transport.broadcast(Channel::Unreliable, &data);
        }
    }

    /// Send an authority message to all peers.
    pub fn send_authority(&mut self, msg: &AuthorityMessage) {
        if let Ok(data) = rmp_serde::to_vec(msg) {
            let _ = self.transport.broadcast(Channel::Reliable, &data);
        }
    }

    /// Number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Whether we're in a multiplayer session.
    pub fn is_multiplayer(&self) -> bool {
        self.transport.peer_count() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopback::loopback_pair;

    #[test]
    fn session_join_and_tick() {
        let (a_transport, _b_transport) = loopback_pair();
        let config = NetworkConfig::local_test();
        let mut session = NetworkSession::new(a_transport, config);

        session.join("test-room").unwrap();
        session.tick();
        assert_eq!(session.tick, 1);
    }

    #[test]
    fn session_send_receive_physics() {
        let (a_transport, b_transport) = loopback_pair();
        let config = NetworkConfig::local_test();

        let mut session_a = NetworkSession::new(a_transport, config.clone());
        let mut session_b = NetworkSession::new(b_transport, config);

        session_a.join("test").unwrap();
        session_b.join("test").unwrap();

        // A sends physics
        session_a.send_physics(vec![BodyStateUpdate {
            body_id: 1,
            position: [10.0, 20.0, 0.0],
            velocity: [1.0, 0.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 0.0],
        }]);

        // B receives on next tick
        session_b.tick();
        assert_eq!(session_b.incoming_physics.len(), 1);
        assert_eq!(session_b.incoming_physics[0].bodies[0].body_id, 1);
        assert_eq!(session_b.incoming_physics[0].bodies[0].position[0], 10.0);
    }

    #[test]
    fn session_send_receive_authority() {
        let (a_transport, b_transport) = loopback_pair();
        let config = NetworkConfig::local_test();

        let mut session_a = NetworkSession::new(a_transport, config.clone());
        let mut session_b = NetworkSession::new(b_transport, config);

        session_a.join("test").unwrap();
        session_b.join("test").unwrap();

        // A claims authority
        session_a.send_authority(&AuthorityMessage::Claim {
            body_ids: vec![1, 2, 3],
            peer_id: 0,
        });

        // B receives
        session_b.tick();
        assert_eq!(session_b.incoming_authority.len(), 1);
        match &session_b.incoming_authority[0] {
            AuthorityMessage::Claim { body_ids, peer_id } => {
                assert_eq!(body_ids, &[1, 2, 3]);
                assert_eq!(*peer_id, 0);
            }
            _ => panic!("Expected Claim"),
        }
    }
}
