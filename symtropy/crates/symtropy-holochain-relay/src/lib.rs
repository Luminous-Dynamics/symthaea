// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Holochain Relay — WebSocket bridge between Symtropy and Holochain conductor.
//!
//! When a Holochain conductor is available, this relay:
//! 1. Connects to the conductor's AdminWebsocket
//! 2. Installs/activates governance, commons, civic hApps
//! 3. Maps game agent IDs to Holochain AgentPubKeys
//! 4. Forwards game actions and compares results to local simulation
//!
//! Two modes:
//! - **Shadow mode**: Log discrepancies between simulation and conductor (default)
//! - **Authoritative mode**: Conductor results override simulation
//!
//! ## Future Implementation
//!
//! The relay will use `tokio-tungstenite` to connect to the Holochain AdminWebsocket
//! on `ws://localhost:4444`. Game events are serialized as JSON-RPC calls matching
//! the Holochain conductor API.

use serde::{Deserialize, Serialize};

/// Relay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayConfig {
    /// Holochain conductor WebSocket URL.
    pub conductor_url: String,
    /// Relay mode.
    pub mode: RelayMode,
    /// Whether to auto-connect on startup.
    pub auto_connect: bool,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            conductor_url: "ws://localhost:4444".to_string(),
            mode: RelayMode::Shadow,
            auto_connect: false,
        }
    }
}

/// Relay operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelayMode {
    /// Log-only: compare simulation to conductor, report discrepancies.
    Shadow,
    /// Conductor-authoritative: override simulation with conductor results.
    Authoritative,
}

/// Game action to forward to the conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameAction {
    /// Cast a governance vote.
    CastVote {
        proposal_id: String,
        voter_id: String,
        phi_score: f64,
        approve: bool,
    },
    /// Execute a TEND exchange.
    TendExchange {
        provider_id: String,
        receiver_id: String,
        amount: i64,
    },
    /// Submit FL gradient.
    SubmitGradient {
        node_id: String,
        gradient: Vec<f32>,
        round: u64,
    },
    /// Update consciousness profile.
    UpdateConsciousness {
        agent_id: String,
        identity: f64,
        reputation: f64,
        community: f64,
        engagement: f64,
    },
}

/// Result from the conductor (or comparison with simulation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayResult {
    /// Whether the action succeeded on the conductor.
    pub success: bool,
    /// Discrepancy between simulation and conductor result.
    pub discrepancy: Option<String>,
    /// Response from the conductor (serialized).
    pub response: Option<String>,
}

/// Relay connection state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected — game uses local simulation only.
    Disconnected,
    /// Connecting to conductor...
    Connecting,
    /// Connected and active.
    Connected,
    /// Connection failed — falling back to simulation.
    Failed(String),
}

/// Relay stub — placeholder for the full WebSocket implementation.
///
/// Currently a no-op. When implemented:
/// 1. `connect()` opens WebSocket to conductor
/// 2. `send_action()` forwards game actions
/// 3. `compare()` checks simulation vs conductor results
pub struct HolochainRelay {
    pub config: RelayConfig,
    pub state: ConnectionState,
    /// Discrepancy count (simulation ≠ conductor).
    pub discrepancy_count: u32,
    /// Total actions forwarded.
    pub total_actions: u32,
}

impl Default for HolochainRelay {
    fn default() -> Self {
        Self {
            config: RelayConfig::default(),
            state: ConnectionState::Disconnected,
            discrepancy_count: 0,
            total_actions: 0,
        }
    }
}

impl HolochainRelay {
    /// Create a new relay with custom config.
    pub fn new(config: RelayConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Check if connected to conductor.
    pub fn is_connected(&self) -> bool {
        self.state == ConnectionState::Connected
    }

    /// Discrepancy rate (0.0 = perfect match, 1.0 = all different).
    pub fn discrepancy_rate(&self) -> f64 {
        if self.total_actions == 0 {
            return 0.0;
        }
        self.discrepancy_count as f64 / self.total_actions as f64
    }

    /// Send a game action to the conductor (stub — no-op until WebSocket implemented).
    pub fn send_action(&mut self, _action: &GameAction) -> RelayResult {
        self.total_actions += 1;

        // Stub: always returns success with no discrepancy
        RelayResult {
            success: true,
            discrepancy: None,
            response: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_default_disconnected() {
        let relay = HolochainRelay::default();
        assert!(!relay.is_connected());
        assert_eq!(relay.state, ConnectionState::Disconnected);
    }

    #[test]
    fn relay_stub_action() {
        let mut relay = HolochainRelay::default();
        let action = GameAction::CastVote {
            proposal_id: "prop-1".into(),
            voter_id: "npc-kael".into(),
            phi_score: 0.6,
            approve: true,
        };
        let result = relay.send_action(&action);
        assert!(result.success);
        assert_eq!(relay.total_actions, 1);
        assert_eq!(relay.discrepancy_count, 0);
    }

    #[test]
    fn relay_discrepancy_rate() {
        let mut relay = HolochainRelay::default();
        relay.total_actions = 100;
        relay.discrepancy_count = 5;
        assert!((relay.discrepancy_rate() - 0.05).abs() < f64::EPSILON);
    }
}
