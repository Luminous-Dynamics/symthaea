// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Peer identification and state tracking.

use symtropy_holochain_relay::ConnectionState;
pub use symtropy_net_core::peer::{PeerId, PeerStateCore};

/// State of a remote peer.
#[derive(Debug, Clone)]
pub struct PeerState {
    /// Core peer fields.
    pub core: PeerStateCore,
    /// Holochain connection state.
    pub connection: ConnectionState,
}

impl PeerState {
    /// Create a local peer.
    pub fn local(id: PeerId, name: impl Into<String>, seed: u64) -> Self {
        Self {
            core: PeerStateCore::local(id, name, seed),
            connection: ConnectionState::Disconnected,
        }
    }

    /// Create a remote peer.
    pub fn remote(id: PeerId, name: impl Into<String>, seed: u64) -> Self {
        Self {
            core: PeerStateCore::remote(id, name, seed),
            connection: ConnectionState::Disconnected,
        }
    }

    /// Peer identifier.
    pub fn id(&self) -> PeerId {
        self.core.id
    }

    /// Display name.
    pub fn name(&self) -> &str {
        &self.core.name
    }

    /// Whether we've heard from this peer recently (within N ticks).
    pub fn is_alive(&self, current_tick: u64, timeout_ticks: u64) -> bool {
        self.core.is_alive(current_tick, timeout_ticks)
    }

    /// Update last seen tick.
    pub fn mark_seen(&mut self, tick: u64) {
        self.core.mark_seen(tick);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_peer_always_alive() {
        let peer = PeerState::local(PeerId(0), "Player", 42);
        assert!(peer.is_alive(1000, 10));
    }

    #[test]
    fn remote_peer_timeout() {
        let mut peer = PeerState::remote(PeerId(1), "Remote", 42);
        peer.mark_seen(100);
        assert!(peer.is_alive(105, 10)); // Within timeout
        assert!(!peer.is_alive(200, 10)); // Past timeout
    }

    #[test]
    fn mark_seen_updates() {
        let mut peer = PeerState::remote(PeerId(1), "Remote", 42);
        assert_eq!(peer.core.last_seen_tick, 0);
        peer.mark_seen(50);
        assert_eq!(peer.core.last_seen_tick, 50);
    }
}
