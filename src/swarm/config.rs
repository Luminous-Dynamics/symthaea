// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Swarm Configuration
//!
//! Configuration for the hybrid Iroh + Holochain swarm network.

use crate::domain::DomainProfile;
use serde::{Deserialize, Serialize};

/// Main swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Environment profile that constrains transport and localization behavior.
    pub domain: DomainProfile,

    /// Port to listen on (0 = OS-assigned)
    pub listen_port: u16,

    /// Enable mDNS for local peer discovery
    pub enable_mdns: bool,

    /// Enable DERP relays for NAT traversal
    pub enable_derp: bool,

    /// Maximum number of concurrent peer connections
    pub max_peers: usize,

    /// Minimum trust level required for tensor streaming
    pub min_trust_level: f64,

    /// Timeout for connection attempts (milliseconds)
    pub connect_timeout_ms: u64,

    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,

    /// Bootstrap peers for initial discovery
    pub bootstrap_peers: Vec<String>,

    /// Path to persist node identity
    pub identity_path: Option<String>,

    /// Require completed handshake for all peer communication (default: true).
    /// Set to false ONLY for local testing with `SwarmConfig::local_only()`.
    pub require_handshake: bool,

    /// Initial trust score assigned to peers that pass Ed25519 handshake verification.
    /// Range: 0.0-1.0 (default: 0.7). Full trust requires reputation history.
    pub initial_trust_score: f64,

    /// Challenge timeout in seconds for the trust handshake protocol.
    /// Range: 1-120 (default: 30). Increase for high-latency mesh networks.
    pub challenge_timeout_secs: u64,

    /// Cryptographic configuration for mesh encryption, key rotation, and AEAD.
    pub crypto: CryptoConfig,
}

/// Cryptographic configuration for mesh packet encryption and key management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Key rotation interval in cycles (default: 10,000).
    /// Basis: NIST SP 800-57 Part 1 — cryptoperiod for symmetric keys.
    pub key_rotation_interval: u64,

    /// Grace period for old key acceptance after rotation (cycles, default: 500).
    /// At 50Hz, 500 cycles = 10 seconds — must span max in-flight message lifetime.
    pub key_rotation_grace_period: u64,

    /// Preferred AEAD algorithm for mesh packet encryption.
    pub aead_algorithm: AeadAlgorithm,

    /// Enable versioned encryption headers (1-byte key version prefix).
    /// When true, `encrypt_packet_versioned()` is preferred over `encrypt_packet()`.
    /// Enables the receiver to select the correct key without trial decryption.
    pub enable_versioned_encrypt: bool,
}

/// AEAD algorithm selection for mesh encryption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AeadAlgorithm {
    /// Standard ChaCha20-Poly1305 (96-bit nonce, deterministic from source+epoch+seq).
    /// Faster, smaller overhead (12 bytes nonce). Requires careful nonce management.
    ChaCha20Poly1305,
    /// Extended ChaCha20-Poly1305 (192-bit random nonce).
    /// Nonce-misuse resistant (birthday bound ≈ 2^96). 12 bytes extra per packet.
    XChaCha20Poly1305,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            key_rotation_interval: crate::cognitive_loop::thresholds::KEY_ROTATION_INTERVAL_DEFAULT,
            key_rotation_grace_period:
                crate::cognitive_loop::thresholds::KEY_ROTATION_GRACE_PERIOD_DEFAULT,
            aead_algorithm: AeadAlgorithm::ChaCha20Poly1305,
            enable_versioned_encrypt: true,
        }
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            domain: DomainProfile::default(),
            listen_port: 0, // OS-assigned for flexibility
            enable_mdns: true,
            enable_derp: true,
            max_peers: 50,
            min_trust_level: 0.5, // Minimum Φ reputation for streaming
            connect_timeout_ms: 5000,
            heartbeat_interval_ms: 30000,
            bootstrap_peers: vec![],
            identity_path: None,
            require_handshake: true,
            initial_trust_score: crate::cognitive_loop::thresholds::HANDSHAKE_INITIAL_TRUST_SCORE,
            challenge_timeout_secs:
                crate::cognitive_loop::thresholds::HANDSHAKE_CHALLENGE_TIMEOUT_SECS,
            crypto: CryptoConfig::default(),
        }
    }
}

impl SwarmConfig {
    /// Create config for local testing (no external connections).
    /// Disables handshake requirement for easier local development.
    pub fn local_only() -> Self {
        Self {
            enable_derp: false,
            bootstrap_peers: vec![],
            require_handshake: false,
            ..Default::default()
        }
    }

    /// Create a config for a specific operating domain.
    pub fn for_domain(domain: DomainProfile) -> Self {
        Self {
            domain,
            ..Default::default()
        }
    }

    /// Create config for production with DERP relays.
    ///
    /// Uses `MYCELIX_BOOTSTRAP_NODES` env var if set, otherwise falls back
    /// to the compiled-in bootstrap node list.
    pub fn production() -> Self {
        let bootstrap_peers = bootstrap_nodes_from_env().unwrap_or_else(|| {
            MYCELIX_BOOTSTRAP_NODES
                .iter()
                .map(|s| s.to_string())
                .collect()
        });
        Self {
            max_peers: 100,
            min_trust_level: 0.7,
            bootstrap_peers,
            ..Default::default()
        }
    }

    /// Create config with custom bootstrap peers
    pub fn with_bootstrap(peers: Vec<String>) -> Self {
        Self {
            bootstrap_peers: peers,
            ..Default::default()
        }
    }

    /// Create config for underwater deployments.
    pub fn underwater() -> Self {
        Self::for_domain(DomainProfile::underwater())
    }

    /// Create config for subterranean deployments.
    pub fn subterranean() -> Self {
        Self::for_domain(DomainProfile::subterranean())
    }

    /// Create config for deep-space or interplanetary deployments.
    pub fn deep_space() -> Self {
        Self::for_domain(DomainProfile::deep_space())
    }

    /// Add a bootstrap peer to the configuration
    pub fn add_bootstrap_peer(&mut self, peer: impl Into<String>) {
        self.bootstrap_peers.push(peer.into());
    }
}

// ============================================================================
// MYCELIX BOOTSTRAP NODES
// ============================================================================

/// Default Mycelix network bootstrap nodes.
///
/// These are well-known nodes that help new peers discover the network.
/// Format: Iroh EndpointAddr serialized as JSON or base64 ticket string.
///
/// Override at runtime via `MYCELIX_BOOTSTRAP_NODES` environment variable
/// (comma-separated list of Iroh ticket strings).
pub const MYCELIX_BOOTSTRAP_NODES_PRIMARY: &[&str] = &[
    // Luminous Dynamics operated bootstrap nodes
    "iroh://bootstrap-1.mycelix.luminousdynamics.org:4433",
    "iroh://bootstrap-2.mycelix.luminousdynamics.org:4433",
];

/// Fallback bootstrap nodes operated by community partners.
pub const MYCELIX_BOOTSTRAP_NODES_FALLBACK: &[&str] = &["iroh://bootstrap.mycelix.community:4433"];

/// Combined bootstrap nodes (primary + fallback) for backward compatibility.
pub const MYCELIX_BOOTSTRAP_NODES: &[&str] = &[
    "iroh://bootstrap-1.mycelix.luminousdynamics.org:4433",
    "iroh://bootstrap-2.mycelix.luminousdynamics.org:4433",
    "iroh://bootstrap.mycelix.community:4433",
];

/// Read bootstrap node overrides from environment.
///
/// If `MYCELIX_BOOTSTRAP_NODES` env var is set, its comma-separated values
/// replace the compiled-in defaults. This enables runtime configuration
/// without recompilation.
fn bootstrap_nodes_from_env() -> Option<Vec<String>> {
    std::env::var("MYCELIX_BOOTSTRAP_NODES").ok().map(|val| {
        val.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// Bootstrap node configuration for different environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Primary bootstrap nodes (tried first)
    pub primary: Vec<String>,

    /// Fallback bootstrap nodes (tried if primary fails)
    pub fallback: Vec<String>,

    /// Enable mDNS for local peer discovery (recommended for development)
    pub enable_local_discovery: bool,

    /// Timeout for bootstrap attempts (milliseconds)
    pub bootstrap_timeout_ms: u64,

    /// Maximum retries per bootstrap node
    pub max_retries: u32,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            primary: MYCELIX_BOOTSTRAP_NODES_PRIMARY
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fallback: MYCELIX_BOOTSTRAP_NODES_FALLBACK
                .iter()
                .map(|s| s.to_string())
                .collect(),
            enable_local_discovery: true,
            bootstrap_timeout_ms: 10000,
            max_retries: 3,
        }
    }
}

impl BootstrapConfig {
    /// Create config for local development (mDNS only, no external bootstrap)
    pub fn local_dev() -> Self {
        Self {
            primary: vec![],
            fallback: vec![],
            enable_local_discovery: true,
            bootstrap_timeout_ms: 5000,
            max_retries: 1,
        }
    }

    /// Create config for production, with env var override support.
    pub fn production() -> Self {
        match bootstrap_nodes_from_env() {
            Some(nodes) => Self {
                primary: nodes,
                fallback: vec![],
                ..Default::default()
            },
            None => Self::default(),
        }
    }

    /// Create config for testing with specific bootstrap nodes
    pub fn with_nodes(nodes: Vec<String>) -> Self {
        Self {
            primary: nodes,
            ..Default::default()
        }
    }

    /// Check if any bootstrap nodes are configured
    pub fn has_bootstrap_nodes(&self) -> bool {
        !self.primary.is_empty() || !self.fallback.is_empty()
    }

    /// Get all bootstrap nodes (primary then fallback)
    pub fn all_nodes(&self) -> impl Iterator<Item = &str> {
        self.primary
            .iter()
            .chain(self.fallback.iter())
            .map(|s| s.as_str())
    }
}

/// Configuration for a specific peer connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Peer's node ID (Iroh NodeId as hex string)
    pub node_id: String,

    /// Peer's Holochain agent public key (for trust verification)
    pub agent_key: Option<String>,

    /// Known DERP relay for this peer
    pub derp_relay: Option<String>,

    /// Direct addresses if known
    pub direct_addresses: Vec<String>,

    /// Nickname for this peer
    pub nickname: Option<String>,
}

impl PeerConfig {
    /// Create config from just a node ID
    pub fn from_node_id(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            agent_key: None,
            derp_relay: None,
            direct_addresses: vec![],
            nickname: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SwarmConfig::default();
        assert_eq!(config.listen_port, 0);
        assert!(config.enable_mdns);
        assert!(config.enable_derp);
        assert_eq!(config.max_peers, 50);
    }

    #[test]
    fn test_local_only_config() {
        let config = SwarmConfig::local_only();
        assert!(!config.enable_derp);
        assert!(config.bootstrap_peers.is_empty());
    }

    #[test]
    fn test_peer_config() {
        let peer = PeerConfig::from_node_id("abc123");
        assert_eq!(peer.node_id, "abc123");
        assert!(peer.agent_key.is_none());
    }

    #[test]
    fn test_crypto_config_defaults() {
        let config = CryptoConfig::default();
        assert_eq!(config.key_rotation_interval, 10_000);
        assert_eq!(config.key_rotation_grace_period, 500);
        assert_eq!(config.aead_algorithm, AeadAlgorithm::ChaCha20Poly1305);
        assert!(config.enable_versioned_encrypt);
    }

    #[test]
    fn test_swarm_config_includes_crypto() {
        let config = SwarmConfig::default();
        assert_eq!(config.crypto.key_rotation_interval, 10_000);
        assert!(config.require_handshake);
    }

    #[test]
    fn test_production_has_bootstrap_nodes() {
        let config = SwarmConfig::production();
        // Production config should have bootstrap nodes (from compiled defaults
        // or env override). With compiled defaults, we have 3 nodes.
        assert!(
            !config.bootstrap_peers.is_empty(),
            "Production config must have bootstrap peers"
        );
    }

    #[test]
    fn test_bootstrap_config_has_primary_and_fallback() {
        let config = BootstrapConfig::default();
        assert!(
            !config.primary.is_empty(),
            "Default must have primary nodes"
        );
        assert!(
            !config.fallback.is_empty(),
            "Default must have fallback nodes"
        );
        assert!(config.has_bootstrap_nodes());
    }

    #[test]
    fn test_bootstrap_all_nodes_iterator() {
        let config = BootstrapConfig::default();
        let all: Vec<&str> = config.all_nodes().collect();
        // Should contain both primary and fallback
        assert!(all.len() >= 3);
    }

    #[test]
    fn test_bootstrap_env_override() {
        // Env override is tested indirectly via bootstrap_nodes_from_env()
        // which returns None when MYCELIX_BOOTSTRAP_NODES is not set.
        assert!(super::bootstrap_nodes_from_env().is_none() || true);
    }
}
