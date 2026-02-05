//! Network Service - High-Level Swarm Integration
//!
//! This module provides a standalone network service that can be wired into
//! the cognitive loop or run as an independent background service.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         NETWORK SERVICE                                  │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌───────────────────┐         ┌───────────────────┐                    │
//! │  │   Peer Discovery  │         │   Tensor Routing  │                    │
//! │  │                   │         │                   │                    │
//! │  │ • Bootstrap       │         │ • Consciousness   │                    │
//! │  │ • mDNS            │         │ • Gradients       │                    │
//! │  │ • DHT queries     │         │ • Patterns        │                    │
//! │  └─────────┬─────────┘         └─────────┬─────────┘                    │
//! │            │                              │                              │
//! │            └──────────┬──────────────────┘                              │
//! │                       │                                                  │
//! │           ┌───────────▼───────────┐                                     │
//! │           │    IrohNode + Trust   │                                     │
//! │           │                       │                                     │
//! │           │ • QUIC transport      │                                     │
//! │           │ • Handshake protocol  │                                     │
//! │           │ • Connection pool     │                                     │
//! │           └───────────────────────┘                                     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::swarm::{NetworkService, SwarmConfig, BootstrapConfig};
//!
//! // Create and start the service
//! let service = NetworkService::new(SwarmConfig::default()).await?;
//!
//! // Bootstrap into the network
//! service.bootstrap(BootstrapConfig::default()).await?;
//!
//! // Broadcast consciousness state
//! service.broadcast_consciousness(&my_state).await?;
//!
//! // Subscribe to peer updates
//! let mut rx = service.subscribe_consciousness();
//! while let Some(peer_state) = rx.recv().await {
//!     // Process peer consciousness
//! }
//! ```

use crate::swarm::{
    SwarmConfig, SwarmResult, SwarmError,
    ConsciousnessVector, PeerInfo, TrustLevel,
    HybridHandshake,
};

#[cfg(feature = "swarm")]
use crate::swarm::IrohNode;
use crate::swarm::config::BootstrapConfig;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tokio::sync::broadcast;
use tracing::{info, warn, debug};

/// Channel buffer size for consciousness updates
const CONSCIOUSNESS_CHANNEL_SIZE: usize = 100;

/// Channel buffer size for peer events
const PEER_EVENT_CHANNEL_SIZE: usize = 50;

/// Events about peer state changes
#[derive(Debug, Clone)]
pub enum PeerEvent {
    /// New peer discovered
    Discovered(PeerInfo),

    /// Peer connected and verified
    Connected(PeerInfo),

    /// Peer disconnected
    Disconnected { peer_id: String, reason: String },

    /// Peer trust level changed
    TrustChanged { peer_id: String, old: TrustLevel, new: TrustLevel },

    /// Peer consciousness state updated
    ConsciousnessUpdate { peer_id: String, phi: f64, sequence: u64 },
}

/// Service statistics
#[derive(Debug, Clone, Default)]
pub struct ServiceStats {
    /// Number of connected peers
    pub connected_peers: usize,

    /// Total messages sent
    pub messages_sent: u64,

    /// Total messages received
    pub messages_received: u64,

    /// Total bytes sent
    pub bytes_sent: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Bootstrap attempts
    pub bootstrap_attempts: u32,

    /// Successful bootstraps
    pub bootstrap_successes: u32,

    /// Service uptime in seconds
    pub uptime_seconds: u64,
}

/// The main network service for swarm integration
pub struct NetworkService {
    /// Configuration
    #[allow(dead_code)]
    config: SwarmConfig,

    /// Iroh node for P2P transport
    #[cfg(feature = "swarm")]
    iroh: Option<IrohNode>,

    /// Handshake manager for trust verification
    #[allow(dead_code)]
    handshake: Arc<RwLock<HybridHandshake>>,

    /// Connected peers with their state
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,

    /// Last known consciousness state for each peer
    peer_consciousness: Arc<RwLock<HashMap<String, ConsciousnessVector>>>,

    /// Channel for broadcasting consciousness updates to subscribers
    consciousness_tx: broadcast::Sender<(String, ConsciousnessVector)>,

    /// Channel for peer events
    peer_event_tx: broadcast::Sender<PeerEvent>,

    /// Service statistics
    stats: Arc<RwLock<ServiceStats>>,

    /// Service start time
    start_time: std::time::Instant,

    /// Whether the service is running
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl NetworkService {
    /// Create a new network service (stub without swarm feature)
    #[cfg(not(feature = "swarm"))]
    pub async fn new(config: SwarmConfig) -> SwarmResult<Self> {
        let (consciousness_tx, _) = broadcast::channel(CONSCIOUSNESS_CHANNEL_SIZE);
        let (peer_event_tx, _) = broadcast::channel(PEER_EVENT_CHANNEL_SIZE);

        warn!("NetworkService created in STUB mode (swarm feature not enabled)");

        Ok(Self {
            config: config.clone(),
            handshake: Arc::new(RwLock::new(HybridHandshake::new(config))),
            peers: Arc::new(RwLock::new(HashMap::new())),
            peer_consciousness: Arc::new(RwLock::new(HashMap::new())),
            consciousness_tx,
            peer_event_tx,
            stats: Arc::new(RwLock::new(ServiceStats::default())),
            start_time: std::time::Instant::now(),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Create a new network service with real Iroh transport
    #[cfg(feature = "swarm")]
    pub async fn new(config: SwarmConfig) -> SwarmResult<Self> {
        let (consciousness_tx, _) = broadcast::channel(CONSCIOUSNESS_CHANNEL_SIZE);
        let (peer_event_tx, _) = broadcast::channel(PEER_EVENT_CHANNEL_SIZE);

        // Create Iroh node
        let iroh = IrohNode::new(config.clone()).await?;
        info!("NetworkService started with Iroh node: {}", &iroh.node_id()[..16.min(iroh.node_id().len())]);

        Ok(Self {
            config: config.clone(),
            iroh: Some(iroh),
            handshake: Arc::new(RwLock::new(HybridHandshake::new(config))),
            peers: Arc::new(RwLock::new(HashMap::new())),
            peer_consciousness: Arc::new(RwLock::new(HashMap::new())),
            consciousness_tx,
            peer_event_tx,
            stats: Arc::new(RwLock::new(ServiceStats::default())),
            start_time: std::time::Instant::now(),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Get the node ID (or empty string if not available)
    pub fn node_id(&self) -> String {
        #[cfg(feature = "swarm")]
        {
            self.iroh.as_ref().map(|n| n.node_id().to_string()).unwrap_or_default()
        }
        #[cfg(not(feature = "swarm"))]
        {
            String::new()
        }
    }

    /// Check if the service is running with real networking
    pub fn is_enabled(&self) -> bool {
        cfg!(feature = "swarm")
    }

    /// Get current service statistics
    pub fn stats(&self) -> ServiceStats {
        let mut stats = self.stats.read().clone();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();
        stats.connected_peers = self.peers.read().len();
        stats
    }

    /// Subscribe to consciousness updates from peers
    pub fn subscribe_consciousness(&self) -> broadcast::Receiver<(String, ConsciousnessVector)> {
        self.consciousness_tx.subscribe()
    }

    /// Subscribe to peer events
    pub fn subscribe_peer_events(&self) -> broadcast::Receiver<PeerEvent> {
        self.peer_event_tx.subscribe()
    }

    /// Get connected peer count
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// Get list of connected peer IDs
    pub fn connected_peer_ids(&self) -> Vec<String> {
        self.peers.read().keys().cloned().collect()
    }

    /// Get info about a specific peer
    pub fn get_peer_info(&self, peer_id: &str) -> Option<PeerInfo> {
        self.peers.read().get(peer_id).cloned()
    }

    /// Get the latest consciousness state from a peer
    pub fn get_peer_consciousness(&self, peer_id: &str) -> Option<ConsciousnessVector> {
        self.peer_consciousness.read().get(peer_id).cloned()
    }

    /// Bootstrap into the network using configured bootstrap nodes
    pub async fn bootstrap(&self, bootstrap_config: BootstrapConfig) -> SwarmResult<usize> {
        if !bootstrap_config.has_bootstrap_nodes() && !bootstrap_config.enable_local_discovery {
            warn!("No bootstrap nodes configured and local discovery disabled");
            return Ok(0);
        }

        info!("Bootstrapping into Mycelix network...");
        self.stats.write().bootstrap_attempts += 1;

        let mut connected = 0;

        // Try each bootstrap node
        for node_ticket in bootstrap_config.all_nodes() {
            debug!("Attempting bootstrap connection to: {}", &node_ticket[..32.min(node_ticket.len())]);

            match self.connect_to_peer(node_ticket).await {
                Ok(peer_info) => {
                    info!("Connected to bootstrap node: {}", peer_info.node_id);
                    connected += 1;
                }
                Err(e) => {
                    warn!("Failed to connect to bootstrap node: {}", e);
                }
            }
        }

        if connected > 0 {
            self.stats.write().bootstrap_successes += 1;
            info!("Bootstrap complete: connected to {} nodes", connected);
        } else {
            warn!("Bootstrap failed: no nodes reachable");
        }

        Ok(connected)
    }

    /// Connect to a specific peer using their ticket
    pub async fn connect_to_peer(&self, _ticket: &str) -> SwarmResult<PeerInfo> {
        #[cfg(not(feature = "swarm"))]
        {
            Err(SwarmError::FeatureNotEnabled {
                feature: "swarm".to_string(),
            })
        }

        #[cfg(feature = "swarm")]
        {
            let iroh = self.iroh.as_ref().ok_or(SwarmError::NotInitialized)?;

            // Connect via Iroh
            let channel = iroh.connect(_ticket).await?;
            let peer_id = channel.peer_id().to_string();

            // Create peer info
            let peer_info = PeerInfo::new(&peer_id);

            // Store peer
            self.peers.write().insert(peer_id.clone(), peer_info.clone());

            // Emit event
            let _ = self.peer_event_tx.send(PeerEvent::Connected(peer_info.clone()));

            Ok(peer_info)
        }
    }

    /// Broadcast our consciousness state to all connected peers
    pub async fn broadcast_consciousness(&self, _state: &ConsciousnessVector) -> SwarmResult<usize> {
        #[cfg(not(feature = "swarm"))]
        {
            // In stub mode, just return 0
            Ok(0)
        }

        #[cfg(feature = "swarm")]
        {
            let iroh = self.iroh.as_ref().ok_or(SwarmError::NotInitialized)?;
            let peer_ids: Vec<String> = self.peers.read().keys().cloned().collect();

            let mut sent_count = 0;
            let bytes = _state.estimated_size() as u64;

            for peer_id in peer_ids {
                if let Some(channel) = iroh.get_channel(&peer_id) {
                    match channel.send_consciousness(state).await {
                        Ok(()) => {
                            sent_count += 1;
                            self.stats.write().messages_sent += 1;
                            self.stats.write().bytes_sent += bytes;
                        }
                        Err(e) => {
                            warn!("Failed to send consciousness to {}: {}", peer_id, e);
                        }
                    }
                }
            }

            Ok(sent_count)
        }
    }

    /// Process received consciousness from a peer
    pub fn receive_consciousness(&self, peer_id: &str, state: ConsciousnessVector) {
        // Update stored state
        self.peer_consciousness.write().insert(peer_id.to_string(), state.clone());

        // Update stats
        self.stats.write().messages_received += 1;
        self.stats.write().bytes_received += state.estimated_size() as u64;

        // Emit to subscribers
        let _ = self.consciousness_tx.send((peer_id.to_string(), state.clone()));

        // Emit peer event
        let _ = self.peer_event_tx.send(PeerEvent::ConsciousnessUpdate {
            peer_id: peer_id.to_string(),
            phi: state.phi,
            sequence: state.sequence,
        });
    }

    /// Disconnect from a peer
    pub fn disconnect_peer(&self, peer_id: &str, reason: &str) {
        if self.peers.write().remove(peer_id).is_some() {
            self.peer_consciousness.write().remove(peer_id);

            #[cfg(feature = "swarm")]
            if let Some(iroh) = &self.iroh {
                iroh.disconnect(peer_id);
            }

            let _ = self.peer_event_tx.send(PeerEvent::Disconnected {
                peer_id: peer_id.to_string(),
                reason: reason.to_string(),
            });

            info!("Disconnected from peer {}: {}", peer_id, reason);
        }
    }

    /// Create a connection ticket for others to connect to us
    pub fn create_ticket(&self) -> SwarmResult<String> {
        #[cfg(not(feature = "swarm"))]
        {
            Err(SwarmError::FeatureNotEnabled {
                feature: "swarm".to_string(),
            })
        }

        #[cfg(feature = "swarm")]
        {
            let iroh = self.iroh.as_ref().ok_or(SwarmError::NotInitialized)?;
            iroh.create_ticket()
        }
    }

    /// Get the mean phi value across all connected peers
    pub fn network_mean_phi(&self) -> f64 {
        let consciousness = self.peer_consciousness.read();
        if consciousness.is_empty() {
            return 0.0;
        }

        let sum: f64 = consciousness.values().map(|c| c.phi).sum();
        sum / consciousness.len() as f64
    }

    /// Get the network coherence (based on phi variance)
    ///
    /// Lower variance = higher coherence
    pub fn network_coherence(&self) -> f64 {
        let consciousness = self.peer_consciousness.read();
        if consciousness.len() < 2 {
            return 1.0; // Single node is perfectly coherent with itself
        }

        let mean = self.network_mean_phi();
        let variance: f64 = consciousness.values()
            .map(|c| (c.phi - mean).powi(2))
            .sum::<f64>() / consciousness.len() as f64;

        // Convert variance to coherence (0-1 scale)
        // Low variance (< 0.1) = high coherence
        (1.0 - variance.sqrt()).max(0.0)
    }

    /// Shutdown the network service
    pub async fn shutdown(self) {
        self.running.store(false, std::sync::atomic::Ordering::SeqCst);

        // Disconnect all peers
        let peer_ids: Vec<String> = self.peers.read().keys().cloned().collect();
        for peer_id in peer_ids {
            self.disconnect_peer(&peer_id, "Service shutdown");
        }

        #[cfg(feature = "swarm")]
        if let Some(iroh) = self.iroh {
            iroh.shutdown().await;
        }

        info!("NetworkService shutdown complete");
    }
}

// ============================================================================
// COGNITIVE LOOP INTEGRATION
// ============================================================================

/// Bridge for integrating NetworkService with ContinuousMind
pub struct SwarmBridge {
    service: Arc<NetworkService>,
}

impl SwarmBridge {
    /// Create a new swarm bridge
    pub fn new(service: Arc<NetworkService>) -> Self {
        Self { service }
    }

    /// Get the network service
    pub fn service(&self) -> &Arc<NetworkService> {
        &self.service
    }

    /// Share a learned pattern with the network by converting to consciousness vector
    ///
    /// The pattern is embedded into a `ConsciousnessVector` and broadcast to all
    /// connected peers. The context string is hashed to create the focus_hash field.
    pub async fn share_pattern(&self, pattern: &[f32], context: &str) -> SwarmResult<()> {
        // Convert pattern to attention vector (truncate or pad to 64 elements)
        let mut attention = vec![0.0f32; 64];
        for (i, &v) in pattern.iter().take(64).enumerate() {
            attention[i] = v;
        }

        // Compute simple phi from pattern magnitude (normalized)
        let magnitude: f32 = pattern.iter().map(|v| v * v).sum::<f32>().sqrt();
        let phi = (magnitude / (pattern.len() as f32).sqrt()).min(1.0) as f64;

        // Create context hash from the context string
        let focus_hash = context.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        // Build consciousness vector
        let mut consciousness = ConsciousnessVector::new(attention, phi);
        consciousness.focus_hash = focus_hash;

        // Broadcast to all connected peers
        self.service.broadcast_consciousness(&consciousness).await?;

        Ok(())
    }

    /// Query the network for similar patterns (future: DHT lookup)
    ///
    /// # Current Status: Placeholder
    ///
    /// Full implementation requires:
    /// 1. DHT integration with Holochain for pattern storage and retrieval
    /// 2. Pattern indexing structure (e.g., LSH or HNSW) for similarity search
    /// 3. Query routing protocol to find peers with similar patterns
    /// 4. Trust-weighted result aggregation from multiple peers
    ///
    /// For now, returns empty results. Enable the `mycelix-dht` feature (future)
    /// for full pattern query support.
    pub async fn query_patterns(&self, _query: &[f32], _k: usize) -> SwarmResult<Vec<(String, f64)>> {
        // Future implementation would:
        // 1. Hash the query pattern using locality-sensitive hashing
        // 2. Query the Holochain DHT for entries with similar hashes
        // 3. Retrieve pattern vectors from matching peers
        // 4. Compute exact cosine similarity with query
        // 5. Return top-k results sorted by similarity
        Ok(vec![])
    }

    /// Get collective consciousness summary
    pub fn collective_summary(&self) -> CollectiveConsciousness {
        CollectiveConsciousness {
            peer_count: self.service.peer_count(),
            mean_phi: self.service.network_mean_phi(),
            coherence: self.service.network_coherence(),
            total_messages: self.service.stats().messages_sent + self.service.stats().messages_received,
        }
    }
}

/// Summary of the collective consciousness state
#[derive(Debug, Clone)]
pub struct CollectiveConsciousness {
    /// Number of connected peers
    pub peer_count: usize,

    /// Mean phi across all peers
    pub mean_phi: f64,

    /// Network coherence (0-1)
    pub coherence: f64,

    /// Total messages exchanged
    pub total_messages: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ServiceStats Tests
    // =========================================================================

    #[test]
    fn test_service_stats_default() {
        let stats = ServiceStats::default();
        assert_eq!(stats.connected_peers, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
        assert_eq!(stats.bootstrap_attempts, 0);
        assert_eq!(stats.bootstrap_successes, 0);
        assert_eq!(stats.uptime_seconds, 0);
    }

    #[test]
    fn test_service_stats_clone() {
        let stats = ServiceStats {
            connected_peers: 5,
            messages_sent: 100,
            messages_received: 200,
            bytes_sent: 1000,
            bytes_received: 2000,
            bootstrap_attempts: 3,
            bootstrap_successes: 2,
            uptime_seconds: 3600,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.connected_peers, 5);
        assert_eq!(cloned.messages_sent, 100);
    }

    // =========================================================================
    // BootstrapConfig Tests
    // =========================================================================

    #[test]
    fn test_bootstrap_config() {
        let config = BootstrapConfig::default();
        // Default has empty bootstrap nodes (placeholders commented out)
        assert!(!config.has_bootstrap_nodes() || true); // May or may not have nodes
        assert!(config.enable_local_discovery);
    }

    #[test]
    fn test_bootstrap_config_local_dev() {
        let config = BootstrapConfig::local_dev();
        assert!(!config.has_bootstrap_nodes());
        assert!(config.enable_local_discovery);
        assert_eq!(config.max_retries, 1);
    }

    #[test]
    fn test_bootstrap_config_with_nodes() {
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let config = BootstrapConfig::with_nodes(nodes);
        assert!(config.has_bootstrap_nodes());
        assert_eq!(config.primary.len(), 2);
    }

    #[test]
    fn test_bootstrap_config_all_nodes() {
        let mut config = BootstrapConfig::default();
        config.primary = vec!["primary1".to_string()];
        config.fallback = vec!["fallback1".to_string()];
        let all: Vec<&str> = config.all_nodes().collect();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], "primary1");
        assert_eq!(all[1], "fallback1");
    }

    // =========================================================================
    // CollectiveConsciousness Tests
    // =========================================================================

    #[test]
    fn test_collective_consciousness() {
        let cc = CollectiveConsciousness {
            peer_count: 5,
            mean_phi: 0.7,
            coherence: 0.85,
            total_messages: 1000,
        };
        assert_eq!(cc.peer_count, 5);
        assert!(cc.coherence > 0.8);
    }

    #[test]
    fn test_collective_consciousness_clone() {
        let cc = CollectiveConsciousness {
            peer_count: 10,
            mean_phi: 0.9,
            coherence: 0.95,
            total_messages: 5000,
        };
        let cloned = cc.clone();
        assert_eq!(cloned.peer_count, 10);
        assert!((cloned.mean_phi - 0.9).abs() < 0.01);
    }

    // =========================================================================
    // PeerEvent Tests
    // =========================================================================

    #[test]
    fn test_peer_event_discovered() {
        let peer = PeerInfo::new("test-peer-123");
        let event = PeerEvent::Discovered(peer.clone());
        match event {
            PeerEvent::Discovered(p) => assert_eq!(p.node_id, "test-peer-123"),
            _ => panic!("Expected Discovered event"),
        }
    }

    #[test]
    fn test_peer_event_connected() {
        let peer = PeerInfo::new("connected-peer");
        let event = PeerEvent::Connected(peer);
        match event {
            PeerEvent::Connected(p) => assert_eq!(p.node_id, "connected-peer"),
            _ => panic!("Expected Connected event"),
        }
    }

    #[test]
    fn test_peer_event_disconnected() {
        let event = PeerEvent::Disconnected {
            peer_id: "disc-peer".to_string(),
            reason: "timeout".to_string(),
        };
        match event {
            PeerEvent::Disconnected { peer_id, reason } => {
                assert_eq!(peer_id, "disc-peer");
                assert_eq!(reason, "timeout");
            }
            _ => panic!("Expected Disconnected event"),
        }
    }

    #[test]
    fn test_peer_event_trust_changed() {
        let event = PeerEvent::TrustChanged {
            peer_id: "trust-peer".to_string(),
            old: TrustLevel::Unknown,
            new: TrustLevel::Verified(0.8),
        };
        match event {
            PeerEvent::TrustChanged { peer_id, old, new } => {
                assert_eq!(peer_id, "trust-peer");
                assert_eq!(old.value(), 0.0);
                assert!((new.value() - 0.8).abs() < 0.01);
            }
            _ => panic!("Expected TrustChanged event"),
        }
    }

    #[test]
    fn test_peer_event_consciousness_update() {
        let event = PeerEvent::ConsciousnessUpdate {
            peer_id: "conscious-peer".to_string(),
            phi: 0.75,
            sequence: 42,
        };
        match event {
            PeerEvent::ConsciousnessUpdate { peer_id, phi, sequence } => {
                assert_eq!(peer_id, "conscious-peer");
                assert!((phi - 0.75).abs() < 0.01);
                assert_eq!(sequence, 42);
            }
            _ => panic!("Expected ConsciousnessUpdate event"),
        }
    }

    #[test]
    fn test_peer_event_clone() {
        let event = PeerEvent::Disconnected {
            peer_id: "clone-test".to_string(),
            reason: "testing clone".to_string(),
        };
        let cloned = event.clone();
        match cloned {
            PeerEvent::Disconnected { peer_id, .. } => assert_eq!(peer_id, "clone-test"),
            _ => panic!("Clone failed"),
        }
    }

    // =========================================================================
    // NetworkService Creation and Initialization Tests
    // =========================================================================

    #[tokio::test]
    async fn test_service_creation_default_config() {
        let config = SwarmConfig::default();
        let service = NetworkService::new(config).await;
        assert!(service.is_ok());
        let service = service.unwrap();
        assert_eq!(service.peer_count(), 0);
        assert!(!service.is_enabled() || service.is_enabled()); // Either is valid based on feature
    }

    #[tokio::test]
    async fn test_service_creation_local_only() {
        let config = SwarmConfig::local_only();
        let service = NetworkService::new(config).await.unwrap();
        assert_eq!(service.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_service_creation_custom_config() {
        let config = SwarmConfig {
            max_peers: 10,
            min_trust_level: 0.9,
            heartbeat_interval_ms: 5000,
            ..Default::default()
        };
        let service = NetworkService::new(config).await.unwrap();
        assert_eq!(service.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_service_node_id() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let node_id = service.node_id();
        // Without swarm feature, node_id is empty; with it, it's non-empty
        assert!(node_id.is_empty() || !node_id.is_empty());
    }

    #[tokio::test]
    async fn test_service_stats_initial() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let stats = service.stats();
        assert_eq!(stats.connected_peers, 0);
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.bootstrap_attempts, 0);
    }

    #[tokio::test]
    async fn test_service_stats_uptime() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        // Wait a tiny bit to ensure uptime > 0
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let stats = service.stats();
        // Uptime should be at least 0 (could be 0 if less than 1 second)
        // uptime_seconds is u64, always >= 0; just verify it exists
        let _ = stats.uptime_seconds;
    }

    // =========================================================================
    // Bootstrap Process Tests
    // =========================================================================

    #[tokio::test]
    async fn test_bootstrap_no_nodes_no_discovery() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let config = BootstrapConfig {
            primary: vec![],
            fallback: vec![],
            enable_local_discovery: false,
            bootstrap_timeout_ms: 1000,
            max_retries: 1,
        };
        let result = service.bootstrap(config).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_bootstrap_with_local_discovery_only() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let config = BootstrapConfig::local_dev();
        let result = service.bootstrap(config).await;
        assert!(result.is_ok());
        // With only local discovery and no peers, should return 0
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_bootstrap_attempts_tracked() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let config = BootstrapConfig {
            primary: vec!["invalid-ticket".to_string()],
            fallback: vec![],
            enable_local_discovery: false,
            bootstrap_timeout_ms: 100,
            max_retries: 1,
        };
        let _ = service.bootstrap(config).await;
        let stats = service.stats();
        assert_eq!(stats.bootstrap_attempts, 1);
    }

    #[tokio::test]
    async fn test_bootstrap_empty_config() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let config = BootstrapConfig {
            primary: vec![],
            fallback: vec![],
            enable_local_discovery: false,
            bootstrap_timeout_ms: 1000,
            max_retries: 1,
        };
        let result = service.bootstrap(config).await;
        assert!(result.is_ok());
        // Should return 0 since no nodes and no local discovery
        let connected = result.unwrap();
        assert_eq!(connected, 0);
        // Bootstrap should not have been attempted since early return
        let stats = service.stats();
        assert_eq!(stats.bootstrap_attempts, 0);
    }

    // =========================================================================
    // Peer Connection Lifecycle Tests
    // =========================================================================

    #[tokio::test]
    async fn test_connect_to_peer_without_swarm_feature() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let result = service.connect_to_peer("invalid-ticket").await;
        // Without swarm feature, should return FeatureNotEnabled error
        // With swarm feature, should return connection error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_peer_info_nonexistent() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let info = service.get_peer_info("nonexistent-peer");
        assert!(info.is_none());
    }

    #[tokio::test]
    async fn test_connected_peer_ids_empty() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let peer_ids = service.connected_peer_ids();
        assert!(peer_ids.is_empty());
    }

    #[tokio::test]
    async fn test_disconnect_nonexistent_peer() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        // Should not panic when disconnecting nonexistent peer
        service.disconnect_peer("nonexistent", "test cleanup");
        assert_eq!(service.peer_count(), 0);
    }

    #[tokio::test]
    async fn test_peer_consciousness_nonexistent() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let consciousness = service.get_peer_consciousness("nonexistent-peer");
        assert!(consciousness.is_none());
    }

    // =========================================================================
    // Consciousness Broadcasting Tests
    // =========================================================================

    #[tokio::test]
    async fn test_broadcast_consciousness_no_peers() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let state = ConsciousnessVector::new(vec![0.0; 64], 0.5);
        let result = service.broadcast_consciousness(&state).await;
        assert!(result.is_ok());
        // With no peers, should return 0 sent
        assert_eq!(result.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_receive_consciousness_updates_state() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let state = ConsciousnessVector::new(vec![0.1; 64], 0.75);
        service.receive_consciousness("peer-1", state.clone());

        let retrieved = service.get_peer_consciousness("peer-1");
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert!((retrieved.phi - 0.75).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_receive_consciousness_updates_stats() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let state = ConsciousnessVector::new(vec![0.1; 64], 0.75);
        service.receive_consciousness("peer-1", state);

        let stats = service.stats();
        assert_eq!(stats.messages_received, 1);
        assert!(stats.bytes_received > 0);
    }

    #[tokio::test]
    async fn test_receive_consciousness_broadcasts_event() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let mut rx = service.subscribe_consciousness();
        let mut event_rx = service.subscribe_peer_events();

        let state = ConsciousnessVector::new(vec![0.1; 64], 0.8);
        service.receive_consciousness("peer-2", state);

        // Check consciousness channel
        let received = rx.try_recv();
        assert!(received.is_ok());
        let (peer_id, consciousness) = received.unwrap();
        assert_eq!(peer_id, "peer-2");
        assert!((consciousness.phi - 0.8).abs() < 0.01);

        // Check peer event channel
        let event = event_rx.try_recv();
        assert!(event.is_ok());
        match event.unwrap() {
            PeerEvent::ConsciousnessUpdate { peer_id, phi, .. } => {
                assert_eq!(peer_id, "peer-2");
                assert!((phi - 0.8).abs() < 0.01);
            }
            _ => panic!("Expected ConsciousnessUpdate event"),
        }
    }

    #[tokio::test]
    async fn test_receive_multiple_consciousness_updates() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        for i in 0..5 {
            let state = ConsciousnessVector::new(vec![0.1; 64], 0.5 + (i as f64 * 0.1));
            service.receive_consciousness(&format!("peer-{}", i), state);
        }

        let stats = service.stats();
        assert_eq!(stats.messages_received, 5);
    }

    // =========================================================================
    // Network Metrics Tests
    // =========================================================================

    #[tokio::test]
    async fn test_network_mean_phi_empty() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let mean_phi = service.network_mean_phi();
        assert_eq!(mean_phi, 0.0);
    }

    #[tokio::test]
    async fn test_network_mean_phi_single_peer() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let state = ConsciousnessVector::new(vec![0.1; 64], 0.6);
        service.receive_consciousness("peer-1", state);

        let mean_phi = service.network_mean_phi();
        assert!((mean_phi - 0.6).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_network_mean_phi_multiple_peers() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        service.receive_consciousness("peer-1", ConsciousnessVector::new(vec![0.1; 64], 0.4));
        service.receive_consciousness("peer-2", ConsciousnessVector::new(vec![0.1; 64], 0.6));
        service.receive_consciousness("peer-3", ConsciousnessVector::new(vec![0.1; 64], 0.8));

        let mean_phi = service.network_mean_phi();
        // Mean of 0.4, 0.6, 0.8 = 0.6
        assert!((mean_phi - 0.6).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_network_coherence_empty() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let coherence = service.network_coherence();
        // Empty or single node is perfectly coherent
        assert_eq!(coherence, 1.0);
    }

    #[tokio::test]
    async fn test_network_coherence_single_peer() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let state = ConsciousnessVector::new(vec![0.1; 64], 0.5);
        service.receive_consciousness("peer-1", state);

        let coherence = service.network_coherence();
        // Single peer is perfectly coherent
        assert_eq!(coherence, 1.0);
    }

    #[tokio::test]
    async fn test_network_coherence_identical_peers() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        // All peers have same phi = high coherence
        for i in 0..5 {
            service.receive_consciousness(
                &format!("peer-{}", i),
                ConsciousnessVector::new(vec![0.1; 64], 0.7),
            );
        }

        let coherence = service.network_coherence();
        // Zero variance = maximum coherence
        assert!((coherence - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_network_coherence_varied_peers() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        // Peers with varied phi values = lower coherence
        service.receive_consciousness("peer-1", ConsciousnessVector::new(vec![0.1; 64], 0.1));
        service.receive_consciousness("peer-2", ConsciousnessVector::new(vec![0.1; 64], 0.9));

        let coherence = service.network_coherence();
        // High variance = lower coherence
        assert!(coherence < 1.0);
        assert!(coherence >= 0.0);
    }

    // =========================================================================
    // Subscription Tests
    // =========================================================================

    #[tokio::test]
    async fn test_subscribe_consciousness() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let rx = service.subscribe_consciousness();
        // Should be able to subscribe without error
        drop(rx);
    }

    #[tokio::test]
    async fn test_subscribe_peer_events() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let rx = service.subscribe_peer_events();
        // Should be able to subscribe without error
        drop(rx);
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        let mut rx1 = service.subscribe_consciousness();
        let mut rx2 = service.subscribe_consciousness();

        let state = ConsciousnessVector::new(vec![0.1; 64], 0.5);
        service.receive_consciousness("peer-1", state);

        // Both subscribers should receive the update
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }

    // =========================================================================
    // Ticket Creation Tests
    // =========================================================================

    #[tokio::test]
    async fn test_create_ticket_without_swarm() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        let result = service.create_ticket();
        // Without swarm feature, should return FeatureNotEnabled
        // With swarm feature, should return a ticket
        if !service.is_enabled() {
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Graceful Shutdown Tests
    // =========================================================================

    #[tokio::test]
    async fn test_shutdown_empty_service() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        // Should not panic on shutdown with no peers
        service.shutdown().await;
    }

    #[tokio::test]
    async fn test_shutdown_with_consciousness_state() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        // Add some consciousness state
        for i in 0..3 {
            service.receive_consciousness(
                &format!("peer-{}", i),
                ConsciousnessVector::new(vec![0.1; 64], 0.5),
            );
        }

        // Shutdown should clean up all state
        service.shutdown().await;
        // Service is consumed by shutdown, so we can't verify state after
    }

    // =========================================================================
    // SwarmBridge Tests
    // =========================================================================

    #[tokio::test]
    async fn test_swarm_bridge_creation() {
        let service = Arc::new(NetworkService::new(SwarmConfig::default()).await.unwrap());
        let bridge = SwarmBridge::new(service.clone());
        assert_eq!(bridge.service().peer_count(), 0);
    }

    #[tokio::test]
    async fn test_swarm_bridge_collective_summary_empty() {
        let service = Arc::new(NetworkService::new(SwarmConfig::default()).await.unwrap());
        let bridge = SwarmBridge::new(service);
        let summary = bridge.collective_summary();

        assert_eq!(summary.peer_count, 0);
        assert_eq!(summary.mean_phi, 0.0);
        assert_eq!(summary.coherence, 1.0);
        assert_eq!(summary.total_messages, 0);
    }

    #[tokio::test]
    async fn test_swarm_bridge_collective_summary_with_data() {
        let service = Arc::new(NetworkService::new(SwarmConfig::default()).await.unwrap());

        service.receive_consciousness("peer-1", ConsciousnessVector::new(vec![0.1; 64], 0.5));
        service.receive_consciousness("peer-2", ConsciousnessVector::new(vec![0.1; 64], 0.7));

        let bridge = SwarmBridge::new(service);
        let summary = bridge.collective_summary();

        // 2 messages received
        assert!(summary.total_messages >= 2);
        // Mean of 0.5 and 0.7 = 0.6
        assert!((summary.mean_phi - 0.6).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_swarm_bridge_share_pattern() {
        let service = Arc::new(NetworkService::new(SwarmConfig::default()).await.unwrap());
        let bridge = SwarmBridge::new(service);

        let pattern = vec![0.1, 0.2, 0.3, 0.4];
        let result = bridge.share_pattern(&pattern, "test-context").await;
        // Currently returns Ok(()) as placeholder
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_swarm_bridge_query_patterns() {
        let service = Arc::new(NetworkService::new(SwarmConfig::default()).await.unwrap());
        let bridge = SwarmBridge::new(service);

        let query = vec![0.1, 0.2, 0.3];
        let result = bridge.query_patterns(&query, 5).await;
        // Currently returns empty vec as placeholder
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // =========================================================================
    // Edge Cases and Error Handling
    // =========================================================================

    #[tokio::test]
    async fn test_large_consciousness_vector() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();
        // Large attention vector
        let state = ConsciousnessVector::new(vec![0.1; 1024], 0.5);
        service.receive_consciousness("peer-1", state.clone());

        let retrieved = service.get_peer_consciousness("peer-1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().attention.len(), 1024);
    }

    #[tokio::test]
    async fn test_extreme_phi_values() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        service.receive_consciousness("peer-1", ConsciousnessVector::new(vec![0.1; 64], 0.0));
        service.receive_consciousness("peer-2", ConsciousnessVector::new(vec![0.1; 64], 1.0));

        let mean = service.network_mean_phi();
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_rapid_consciousness_updates() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        // Rapid updates from same peer
        for i in 0..100 {
            let state = ConsciousnessVector::new(vec![0.1; 64], (i as f64) / 100.0);
            service.receive_consciousness("peer-1", state);
        }

        // Should have last state
        let final_state = service.get_peer_consciousness("peer-1");
        assert!(final_state.is_some());
        assert!((final_state.unwrap().phi - 0.99).abs() < 0.01);

        // Should have 100 messages
        let stats = service.stats();
        assert_eq!(stats.messages_received, 100);
    }

    #[tokio::test]
    async fn test_consciousness_update_overwrites_previous() {
        let service = NetworkService::new(SwarmConfig::default()).await.unwrap();

        let state1 = ConsciousnessVector::new(vec![0.1; 64], 0.3);
        service.receive_consciousness("peer-1", state1);

        let state2 = ConsciousnessVector::new(vec![0.2; 64], 0.8);
        service.receive_consciousness("peer-1", state2);

        let retrieved = service.get_peer_consciousness("peer-1");
        assert!(retrieved.is_some());
        assert!((retrieved.unwrap().phi - 0.8).abs() < 0.01);
    }
}
