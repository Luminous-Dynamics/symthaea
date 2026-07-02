// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Peer identification and core networking types.

use serde::{Deserialize, Serialize};

/// Unique peer identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PeerId(pub u64);

/// State of a remote peer (core fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerStateCore {
    /// Peer identifier.
    pub id: PeerId,
    /// Display name.
    pub name: String,
    /// Last tick we received from this peer.
    pub last_seen_tick: u64,
    /// Whether this peer is the local player.
    pub is_local: bool,
    /// Shared RNG seed for deterministic lockstep.
    pub shared_seed: u64,
}

impl PeerStateCore {
    /// Create a local peer.
    pub fn local(id: PeerId, name: impl Into<String>, seed: u64) -> Self {
        Self {
            id,
            name: name.into(),
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
