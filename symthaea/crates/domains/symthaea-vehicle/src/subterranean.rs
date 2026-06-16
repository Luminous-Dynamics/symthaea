// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subterranean relay decisions and survey anchor types.
//!
//! Minimal stubs to support training.rs — a full cave-relay mesh protocol
//! is future work.

use serde::{Deserialize, Serialize};

/// A survey anchor — a known-position beacon for underground localization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurveyAnchor {
    pub anchor_id: String,
    pub position_m: [f64; 3],
    pub admitted: bool,
}

/// A tunnel relay node spec.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TunnelRelayNode {
    pub node_id: String,
    pub position_m: [f64; 3],
}

/// Relay priority for store-and-forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RelayPriority {
    #[default]
    Normal,
    High,
    Emergency,
}

/// Relay decision output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaveRelayDecision {
    /// Forward immediately.
    Forward,
    /// Buffer and forward later (blackout or congestion).
    BufferAndForward,
    /// Drop the message.
    Drop,
}

/// Subterranean mesh relay bridge.
#[derive(Debug, Clone, Default)]
pub struct SubterraneanBridge {
    total_forwards: usize,
    total_buffered: usize,
}

impl SubterraneanBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide how to route a message to the given relay node.
    pub fn relay_decision(
        &self,
        _node: &TunnelRelayNode,
        _priority: RelayPriority,
        blackout: bool,
    ) -> CaveRelayDecision {
        if blackout {
            CaveRelayDecision::BufferAndForward
        } else {
            CaveRelayDecision::Forward
        }
    }

    pub fn total_forwards(&self) -> usize {
        self.total_forwards
    }

    pub fn total_buffered(&self) -> usize {
        self.total_buffered
    }
}
