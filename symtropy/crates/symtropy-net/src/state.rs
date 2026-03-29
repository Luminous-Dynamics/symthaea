// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! State authority classification and syncable state types.

/// What kind of authority governs a piece of state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateAuthority {
    /// Fully local: camera, UI, input. Never replicated.
    Local,
    /// Eventually consistent: agent positions, inventory.
    /// Sent via direct P2P messaging. Stale data accepted.
    Replicated,
    /// Consensus-required: governance votes, TEND transactions.
    /// Goes through Holochain DHT for consistency.
    Consensus,
}

/// A piece of state that can be synced across peers.
#[derive(Debug, Clone)]
pub struct SyncableState {
    /// Unique key for this state.
    pub key: String,
    /// Authority level.
    pub authority: StateAuthority,
    /// Serialized value.
    pub value: Vec<u8>,
    /// Tick when this was last updated.
    pub last_updated: u64,
    /// Which peer last wrote this state.
    pub owner: PeerId,
}

use crate::peer::PeerId;

impl SyncableState {
    /// Create a new local state entry.
    pub fn local(key: impl Into<String>, owner: PeerId) -> Self {
        Self {
            key: key.into(),
            authority: StateAuthority::Local,
            value: Vec::new(),
            last_updated: 0,
            owner,
        }
    }

    /// Create a new replicated state entry.
    pub fn replicated(key: impl Into<String>, owner: PeerId) -> Self {
        Self {
            key: key.into(),
            authority: StateAuthority::Replicated,
            value: Vec::new(),
            last_updated: 0,
            owner,
        }
    }

    /// Create a new consensus state entry.
    pub fn consensus(key: impl Into<String>, owner: PeerId) -> Self {
        Self {
            key: key.into(),
            authority: StateAuthority::Consensus,
            value: Vec::new(),
            last_updated: 0,
            owner,
        }
    }

    /// Whether this state should be sent to the Holochain DHT.
    pub fn requires_dht(&self) -> bool {
        self.authority == StateAuthority::Consensus
    }

    /// Whether this state should be sent via P2P messaging.
    pub fn requires_p2p(&self) -> bool {
        self.authority == StateAuthority::Replicated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_authority_classification() {
        let local = SyncableState::local("camera_pos", PeerId(0));
        assert!(!local.requires_dht());
        assert!(!local.requires_p2p());

        let replicated = SyncableState::replicated("player_pos", PeerId(0));
        assert!(!replicated.requires_dht());
        assert!(replicated.requires_p2p());

        let consensus = SyncableState::consensus("vote", PeerId(0));
        assert!(consensus.requires_dht());
        assert!(!consensus.requires_p2p());
    }
}
