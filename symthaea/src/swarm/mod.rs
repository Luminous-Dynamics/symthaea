// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

mod capability_card;
mod checkpoint;
mod config;
mod error;
pub mod federated_cfc;
mod federated_network;
mod holochain;
#[cfg(feature = "mycelix")]
pub mod hybrid_bft;
mod hyperfeel;
#[cfg(feature = "mesh")]
pub mod mesh;
#[cfg(feature = "liquid-mamba")]
pub mod projection_federated;
mod service;
mod types;

// Iroh and handshake modules compile always (with stub implementations when feature disabled)
pub(crate) mod attestation;
pub(crate) mod handshake;
mod iroh;
#[cfg(feature = "pqc-handshake")]
pub mod pqc_handshake;

// Sovereign Inoculation: Trust + Name + Social
#[cfg(feature = "mesh")]
pub mod consciousness_certificate;
#[cfg(feature = "mesh")]
pub mod name_resolver;
#[cfg(feature = "mesh")]
pub mod resonance_graph;
#[cfg(feature = "mesh")]
pub mod web_of_trust;

/// Holon QUIC transport for RDP datagrams + reliable control streams.
#[cfg(feature = "holon-quic")]
pub mod quic_transport;
/// Adaptive quality engine (AIMD + consciousness tiers).
pub mod rdp_adaptive;
/// Audio capture and playback with consciousness-gated quality.
pub mod rdp_audio;
/// Behavioral biometric continuous authentication via HDC.
pub mod rdp_behavioral_auth;
/// Screen capture abstraction (X11 SHM, Wayland PipeWire, test stubs).
pub mod rdp_capture;
/// Sovereign RDP client (frame consumer + input forwarder).
pub mod rdp_client;
/// Clipboard synchronization (bidirectional, consciousness-gated, scrubbed).
pub mod rdp_clipboard;
/// HDC patch quantization codec for RDP.
pub mod rdp_codec;
/// Encrypted file transfer with BLAKE3 integrity.
pub mod rdp_file_transfer;
/// Governance-gated session approval for sensitive support sessions.
pub mod rdp_governance;
/// Soma↔Holon RDP bridge for phone remote support.
pub mod rdp_holon_bridge;
/// Input injection (X11 XTest, Wayland virtual input, logging/noop).
pub mod rdp_input;
/// PQC-secured remote display protocol.
pub mod rdp_protocol;
/// RDP Protocol v3 extensions (Audio, Chat, RemoteExec, Clipboard).
pub mod rdp_protocol_ext;
/// PQC-encrypted session recording and playback.
pub mod rdp_recording;
/// Remote command execution with consciousness + MFDI gating.
pub mod rdp_remote_exec;
/// Egui renderer for RDP client.
pub mod rdp_render_egui;
/// Sovereign RDP server (tick-based frame producer).
pub mod rdp_server;
/// PQC-authenticated session lifecycle for RDP.
pub mod rdp_session;
/// Support ticket ↔ RDP session bridge with invite tokens and governance.
pub mod rdp_support_bridge;
/// Iroh QUIC transport layer (Handle/Actor pattern, stream multiplexing).
pub mod rdp_transport;
/// Unattended access daemon (MFDI-gated, time-bounded, audit-logged).
pub mod rdp_unattended;
/// Binary wire envelope for RDP (bincode + ChaCha20-Poly1305 via packet_crypto).
pub mod rdp_wire;
/// Sliding-window replay protection for AEAD-sealed streams (RDP + future mesh).
pub mod replay_window;

// Re-exports
pub use config::{
    AeadAlgorithm, BootstrapConfig, CryptoConfig, PeerConfig, SwarmConfig, MYCELIX_BOOTSTRAP_NODES,
};
pub use error::{SwarmError, SwarmResult};
pub use types::{
    AffectiveSync, ConnectionState, ConnectionTicket, ConsciousnessVector, PeerInfo,
    SecurityTelemetry, SwarmMessage, TensorPayload, TensorType, TrustLevel,
};

// Hyperfeel - synthetic mirror neurons
pub use hyperfeel::{
    AffectiveState, EmotionLabel, Hyperfeel, HyperfeelConfig, HyperfeelStats, PeerAffect,
    SwarmCoherence, SwarmHealth, SwarmStatus,
};

// Holochain Cortex - trust and identity
pub use holochain::{
    AgentInfo, AgentPubKey, CortexError, CortexStats, EpigeneticInsight, HolochainConfig,
    HolochainCortex, SignedChallenge, TrustVerificationResult,
};

// Iroh types (stub when feature disabled, real when enabled)
pub use iroh::{
    IrohBridgeActor, IrohBridgeHandle, IrohChannel, IrohNode, StreamConfig, TensorStream,
};

// Handshake types
pub use handshake::{HandshakeError, HandshakeResult, HybridHandshake, SwarmMessageExt};

// Consciousness attestation — signed ConsciousnessVector proofs
pub use attestation::{AttestationManager, AttestedConsciousnessVector};

// Network Service - high-level swarm integration
pub use service::{CollectiveConsciousness, NetworkService, PeerEvent, ServiceStats, SwarmBridge};

// Federated CfC Learning - trust-weighted gradient aggregation
pub use federated_cfc::{DifferentialPrivacyConfig, FederatedAggregator, GradientMessage};

// Federated Network Communication - channel and TCP backends
pub use federated_network::{
    create_test_network, CoordinatorEvent, CoordinatorStats, FederatedCoordinator,
    FederatedMessage, FederatedNetworkConfig, FederatedNode, LocalChannelBackend, NetworkBackend,
    NetworkError, NetworkResult, NodeAddress, TcpBackend,
};

// Capability Card - BLAKE3-hashed self-description for peer discovery
pub use capability_card::{CapabilityCard, CardStats};
mod reputation_bridge;
pub use reputation_bridge::{ReputationBridge, VouchDecision};
mod topological_handshake;
pub use topological_handshake::{evaluate_compatibility, HandshakeConfig};

// Federated Learning Checkpointing - fault-tolerant distributed training
pub use checkpoint::{
    process_node_rejoin, recover_coordinator, BufferedGradient, CheckpointConfig,
    CheckpointManager, CheckpointedConfig, FederatedCheckpoint, IncrementalCheckpoint,
    NodeCheckpoint, NodeRejoinRequest, NodeRejoinResponse, NodeState, RecoveryResult,
};

// ============================================================================
// SWARM NODE - Main Entry Point
// ============================================================================

/// Main swarm node combining Iroh transport with Holochain trust
#[allow(dead_code)] // RESERVED(future): peer-to-peer swarm node
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
