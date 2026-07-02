// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Network configuration — signaling, STUN/TURN, transport settings.
//!
//! All URLs are configurable so game studios deploying Symtropy
//! can swap in their own infrastructure. Defaults use Holo's free
//! public servers.

use serde::{Deserialize, Serialize};

/// Network configuration for Symtropy multiplayer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// WebRTC signaling server URL.
    /// Default: Holo's free public signaling server.
    pub signal_url: String,

    /// STUN server URL for NAT traversal.
    /// Default: Google's free public STUN server.
    pub stun_url: String,

    /// Optional TURN server URL for relay fallback.
    /// Only needed when players are behind symmetric NAT.
    /// Default: None (no relay — direct P2P only).
    pub turn_url: Option<String>,

    /// Optional TURN credentials.
    pub turn_username: Option<String>,
    pub turn_credential: Option<String>,

    /// Maximum peers to connect to simultaneously.
    pub max_peers: usize,

    /// Physics state send rate (Hz). Higher = smoother but more bandwidth.
    pub send_rate_hz: u32,

    /// Authority transfer timeout (ms). If a peer doesn't respond
    /// within this time, their bodies become unclaimed.
    pub authority_timeout_ms: u64,

    /// Room/lobby identifier for matchmaking.
    pub room_id: Option<String>,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            signal_url: std::env::var("SYMTROPY_SIGNAL_URL")
                .unwrap_or_else(|_| "wss://signal.holo.host".to_string()),
            stun_url: std::env::var("SYMTROPY_STUN_URL")
                .unwrap_or_else(|_| "stun:stun.l.google.com:19302".to_string()),
            turn_url: std::env::var("SYMTROPY_TURN_URL").ok(),
            turn_username: std::env::var("SYMTROPY_TURN_USER").ok(),
            turn_credential: std::env::var("SYMTROPY_TURN_CRED").ok(),
            max_peers: 8,
            send_rate_hz: 20,
            authority_timeout_ms: 5000,
            room_id: None,
        }
    }
}

impl NetworkConfig {
    /// Create a config for local testing (no real signaling).
    pub fn local_test() -> Self {
        Self {
            signal_url: "ws://localhost:9001".to_string(),
            stun_url: "stun:stun.l.google.com:19302".to_string(),
            turn_url: None,
            turn_username: None,
            turn_credential: None,
            max_peers: 4,
            send_rate_hz: 20,
            authority_timeout_ms: 3000,
            room_id: Some("test-room".to_string()),
        }
    }

    /// Create a config pointing at self-hosted infrastructure.
    pub fn self_hosted(signal: &str, stun: &str) -> Self {
        Self {
            signal_url: signal.to_string(),
            stun_url: stun.to_string(),
            ..Self::default()
        }
    }
}
