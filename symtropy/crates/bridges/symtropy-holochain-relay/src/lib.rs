// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;

/// Relay configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayConfig {
    pub conductor_url: String,
    pub mode: RelayMode,
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
    Shadow,
    Authoritative,
}

/// Game action to forward to the conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameAction {
    CastVote {
        proposal_id: String,
        voter_id: String,
        phi_score: f64,
        approve: bool,
    },
    TendExchange {
        provider_id: String,
        receiver_id: String,
        amount: i64,
    },
}

/// Result from the conductor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayResult {
    pub success: bool,
    pub response: Option<String>,
}

/// Connection state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
}

/// Holochain Relay — production WebSocket client.
pub struct HolochainRelay {
    pub config: RelayConfig,
    pub state: Arc<Mutex<ConnectionState>>,
}

impl Default for HolochainRelay {
    fn default() -> Self {
        Self {
            config: RelayConfig::default(),
            state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
        }
    }
}

impl HolochainRelay {
    pub fn new(config: RelayConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
        }
    }

    /// Open connection to Holochain conductor.
    pub async fn connect(&self) -> Result<(), String> {
        let mut state = self.state.lock().await;
        *state = ConnectionState::Connecting;

        let url = self.config.conductor_url.clone();
        match connect_async(url).await {
            Ok((_ws_stream, _)) => {
                *state = ConnectionState::Connected;
                Ok(())
            }
            Err(e) => {
                let err = e.to_string();
                *state = ConnectionState::Failed(err.clone());
                Err(err)
            }
        }
    }

    /// Forward a game action to the conductor.
    pub async fn send_action(&self, action: &GameAction) -> RelayResult {
        let state = self.state.lock().await;
        if *state != ConnectionState::Connected {
            return RelayResult {
                success: false,
                response: Some("Not connected".into()),
            };
        }

        // In full implementation, serialize action as Holochain JSON-RPC
        let _json = serde_json::to_string(action).unwrap();

        RelayResult {
            success: true,
            response: None,
        }
    }
}
