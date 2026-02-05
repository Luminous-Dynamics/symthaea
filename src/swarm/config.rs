//! Swarm Configuration
//!
//! Configuration for the hybrid Iroh + Holochain swarm network.

use serde::{Deserialize, Serialize};

/// Main swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
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
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            listen_port: 0, // OS-assigned for flexibility
            enable_mdns: true,
            enable_derp: true,
            max_peers: 50,
            min_trust_level: 0.5, // Minimum Φ reputation for streaming
            connect_timeout_ms: 5000,
            heartbeat_interval_ms: 30000,
            bootstrap_peers: vec![],
            identity_path: None,
        }
    }
}

impl SwarmConfig {
    /// Create config for local testing (no external connections)
    pub fn local_only() -> Self {
        Self {
            enable_derp: false,
            bootstrap_peers: vec![],
            ..Default::default()
        }
    }

    /// Create config for production with DERP relays
    pub fn production() -> Self {
        Self {
            max_peers: 100,
            min_trust_level: 0.7,
            bootstrap_peers: MYCELIX_BOOTSTRAP_NODES.iter().map(|s| s.to_string()).collect(),
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

    /// Add a bootstrap peer to the configuration
    pub fn add_bootstrap_peer(&mut self, peer: impl Into<String>) {
        self.bootstrap_peers.push(peer.into());
    }
}

// ============================================================================
// MYCELIX BOOTSTRAP NODES
// ============================================================================

/// Default Mycelix network bootstrap nodes
///
/// These are well-known nodes that help new peers discover the network.
/// Format: Iroh EndpointAddr serialized as JSON or base64 ticket string.
///
/// In production, these would be operated by trusted parties in the Mycelix
/// network. For development, use local nodes or test infrastructure.
pub const MYCELIX_BOOTSTRAP_NODES: &[&str] = &[
    // Development/Testing bootstrap nodes (localhost)
    // These are placeholders - replace with real bootstrap node tickets
    // "iroh://localhost:4433/symthaea",

    // Luminous Dynamics operated bootstrap (future)
    // "iroh://bootstrap-1.mycelix.luminousdynamics.org:4433",
    // "iroh://bootstrap-2.mycelix.luminousdynamics.org:4433",

    // Community-operated bootstrap nodes (future)
    // "iroh://bootstrap.mycelix.community:4433",
];

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
            primary: MYCELIX_BOOTSTRAP_NODES.iter().map(|s| s.to_string()).collect(),
            fallback: vec![],
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
        self.primary.iter().chain(self.fallback.iter()).map(|s| s.as_str())
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
}
