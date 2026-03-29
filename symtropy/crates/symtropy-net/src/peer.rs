//! Peer identification and state tracking.

use symtropy_holochain_relay::ConnectionState;

/// Unique peer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PeerId(pub u64);

/// State of a remote peer.
#[derive(Debug, Clone)]
pub struct PeerState {
    /// Peer identifier.
    pub id: PeerId,
    /// Display name.
    pub name: String,
    /// Holochain connection state.
    pub connection: ConnectionState,
    /// Last tick we received from this peer.
    pub last_seen_tick: u64,
    /// Whether this peer is the local player.
    pub is_local: bool,
    /// Shared RNG seed for deterministic lockstep.
    pub shared_seed: u64,
}

impl PeerState {
    /// Create a local peer.
    pub fn local(id: PeerId, name: impl Into<String>, seed: u64) -> Self {
        Self {
            id,
            name: name.into(),
            connection: ConnectionState::Disconnected,
            last_seen_tick: 0,
            is_local: true,
            shared_seed: seed,
        }
    }

    /// Create a remote peer.
    pub fn remote(id: PeerId, name: impl Into<String>, seed: u64) -> Self {
        Self {
            id,
            name: name.into(),
            connection: ConnectionState::Disconnected,
            last_seen_tick: 0,
            is_local: false,
            shared_seed: seed,
        }
    }

    /// Whether we've heard from this peer recently (within N ticks).
    pub fn is_alive(&self, current_tick: u64, timeout_ticks: u64) -> bool {
        self.is_local || current_tick.saturating_sub(self.last_seen_tick) < timeout_ticks
    }

    /// Update last seen tick.
    pub fn mark_seen(&mut self, tick: u64) {
        self.last_seen_tick = tick;
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
        assert_eq!(peer.last_seen_tick, 0);
        peer.mark_seen(50);
        assert_eq!(peer.last_seen_tick, 50);
    }
}
