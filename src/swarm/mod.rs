//! Swarm Intelligence Module - Hybrid Iroh + Holochain Architecture
//!
//! This module implements the "Nervous System Pattern" for distributed
//! consciousness coordination:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    SYMTHAEA NERVOUS SYSTEM                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌───────────────────┐         ┌───────────────────┐               │
//! │  │     CORTEX        │         │     SYNAPSE       │               │
//! │  │   (Holochain)     │         │    (Iroh)         │               │
//! │  │                   │         │                   │               │
//! │  │ • Identity        │         │ • Real-time       │               │
//! │  │ • Reputation (Φ)  │         │ • Tensor streams  │               │
//! │  │ • Long-term memory│         │ • Ephemeral state │               │
//! │  │ • Trust graph     │         │ • <50ms latency   │               │
//! │  │ • Access keys     │         │ • QUIC channels   │               │
//! │  └────────┬──────────┘         └──────────┬────────┘               │
//! │           │                               │                         │
//! │           └───────────┬───────────────────┘                         │
//! │                       │                                             │
//! │           ┌───────────▼───────────┐                                 │
//! │           │   HYBRID HANDSHAKE    │                                 │
//! │           │                       │                                 │
//! │           │ 1. Check trust (HC)   │                                 │
//! │           │ 2. Exchange ticket    │                                 │
//! │           │ 3. Open Iroh channel  │                                 │
//! │           │ 4. Stream tensors     │                                 │
//! │           │ 5. Persist summary    │                                 │
//! │           └───────────────────────┘                                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Architecture Decision
//!
//! We use a hybrid approach because:
//!
//! - **Holochain** excels at agent-centric integrity (identity, reputation, audit)
//!   but is too slow for real-time tensor streaming (gossip takes seconds)
//!
//! - **Iroh** excels at low-latency QUIC streams (<50ms) with excellent NAT
//!   traversal, but has no opinion on trust or validation
//!
//! Together they form a nervous system where:
//! - Holochain is the Hippocampus (long-term memory, trust)
//! - Iroh is the Axonal network (fast signal propagation)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::swarm::{SwarmNode, SwarmConfig};
//!
//! let config = SwarmConfig::default();
//! let node = SwarmNode::new(config).await?;
//!
//! // Connect to peer with trust verification
//! let peer_id = "...";
//! let channel = node.connect_trusted(peer_id).await?;
//!
//! // Stream consciousness vectors in real-time
//! channel.send_tensor(&attention_vector).await?;
//! let peer_state = channel.recv_tensor().await?;
//! ```

mod config;
mod error;
mod types;
mod hyperfeel;
mod holochain;
mod service;
mod federated_cfc;
mod federated_network;
mod checkpoint;

// Iroh and handshake modules compile always (with stub implementations when feature disabled)
mod iroh;
mod handshake;

// Re-exports
pub use config::{SwarmConfig, PeerConfig, BootstrapConfig, MYCELIX_BOOTSTRAP_NODES};
pub use error::{SwarmError, SwarmResult};
pub use types::{
    SwarmMessage, ConsciousnessVector, TensorPayload, TensorType,
    PeerInfo, ConnectionTicket, TrustLevel, ConnectionState, AffectiveSync,
};

// Hyperfeel - synthetic mirror neurons
pub use hyperfeel::{
    Hyperfeel, HyperfeelConfig, HyperfeelStats,
    AffectiveState, EmotionLabel, SwarmCoherence,
    SwarmHealth, SwarmStatus, PeerAffect,
};

// Holochain Cortex - trust and identity
pub use holochain::{
    HolochainCortex, HolochainConfig, CortexStats, CortexError,
    AgentPubKey, AgentInfo, SignedChallenge, TrustVerificationResult,
};

// Iroh types (stub when feature disabled, real when enabled)
pub use iroh::{IrohNode, IrohChannel, TensorStream, StreamConfig};

// Handshake types
pub use handshake::{HybridHandshake, HandshakeResult, HandshakeError, SwarmMessageExt};

// Network Service - high-level swarm integration
pub use service::{NetworkService, ServiceStats, PeerEvent, SwarmBridge, CollectiveConsciousness};

// Federated CfC Learning - trust-weighted gradient aggregation
pub use federated_cfc::{
    FederatedAggregator, GradientMessage, DifferentialPrivacyConfig,
};

// Federated Network Communication - channel and TCP backends
pub use federated_network::{
    FederatedNetworkConfig, FederatedMessage, FederatedNode, NodeAddress,
    NetworkBackend, NetworkResult, NetworkError,
    LocalChannelBackend, TcpBackend,
    FederatedCoordinator, CoordinatorStats, CoordinatorEvent,
    create_test_network,
};

// Federated Learning Checkpointing - fault-tolerant distributed training
pub use checkpoint::{
    FederatedCheckpoint, CheckpointedConfig, NodeState, BufferedGradient,
    IncrementalCheckpoint, NodeCheckpoint,
    CheckpointConfig, CheckpointManager,
    RecoveryResult, NodeRejoinRequest, NodeRejoinResponse,
    recover_coordinator, process_node_rejoin,
};

// ============================================================================
// SWARM NODE - Main Entry Point
// ============================================================================

/// Main swarm node combining Iroh transport with Holochain trust
#[allow(dead_code)] // Fields reserved for full implementation
pub struct SwarmNode {
    config: SwarmConfig,
    #[cfg(feature = "swarm")]
    iroh: Option<IrohNode>,
    peers: std::collections::HashMap<String, PeerInfo>,
}

impl SwarmNode {
    /// Create a new swarm node (stub when feature disabled)
    pub fn new(config: SwarmConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "swarm")]
            iroh: None,
            peers: std::collections::HashMap::new(),
        }
    }

    /// Get the node's configuration
    pub fn config(&self) -> &SwarmConfig {
        &self.config
    }

    /// Get connected peers
    pub fn peers(&self) -> &std::collections::HashMap<String, PeerInfo> {
        &self.peers
    }

    /// Check if swarm feature is enabled
    pub fn is_enabled() -> bool {
        cfg!(feature = "swarm")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_node_creation() {
        let config = SwarmConfig::default();
        let node = SwarmNode::new(config);
        assert!(node.peers().is_empty());
    }

    #[test]
    fn test_swarm_config_defaults() {
        let config = SwarmConfig::default();
        assert_eq!(config.listen_port, 0); // OS-assigned
        assert!(config.enable_mdns);
    }
}
