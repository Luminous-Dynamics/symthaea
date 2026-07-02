// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! State authority classification and syncable state types.

use crate::peer::PeerId;
use serde::{Deserialize, Serialize};

/// What kind of authority governs a piece of state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateAuthority {
    /// Fully local: camera, UI, input. Never replicated.
    Local,
    /// Eventually consistent: agent positions, inventory.
    /// Sent via direct P2P messaging. Stale data accepted.
    Replicated,
    /// Consensus-required: governance votes, TEND transactions.
    /// Goes through a distributed ledger (e.g. Holochain DHT) for consistency.
    Consensus,
}

/// A piece of state that can be synced across peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Whether this state should be sent to a consensus DHT.
    pub fn requires_dht(&self) -> bool {
        self.authority == StateAuthority::Consensus
    }

    /// Whether this state should be sent via P2P messaging.
    pub fn requires_p2p(&self) -> bool {
        self.authority == StateAuthority::Replicated
    }
}
