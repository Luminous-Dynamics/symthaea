// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated learning coordinator for multi-node synchronization.

use crate::swarm::federated_cfc::FederatedAggregator;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{broadcast, oneshot};
use tracing::{debug, info, warn};

use super::local_backend::LocalChannelBackend;
use super::types::{
    FederatedMessage, FederatedNetworkConfig, FederatedNode, NetworkBackend, NetworkError,
    NetworkResult, NodeAddress,
};

// ============================================================================
// COORDINATOR TYPES
// ============================================================================

/// Statistics about the federated network
#[derive(Debug, Clone, Default)]
pub struct CoordinatorStats {
    /// Number of active nodes
    pub active_nodes: usize,

    /// Current synchronization round
    pub current_round: u64,

    /// Total gradients received
    pub gradients_received: u64,

    /// Total model updates broadcast
    pub model_updates_broadcast: u64,

    /// Total heartbeats received
    pub heartbeats_received: u64,

    /// Number of failed send attempts
    pub send_failures: u64,

    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Event broadcast by the coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// New node joined
    NodeJoined {
        node_id: [u8; 32],
        address: NodeAddress,
    },

    /// Node left or was removed
    NodeLeft { node_id: [u8; 32], reason: String },

    /// Synchronization round completed
    RoundCompleted { round: u64, participants: usize },

    /// Model updated
    ModelUpdated { round: u64, weights_dim: usize },

    /// Error occurred
    Error { message: String },
}

// ============================================================================
// FEDERATED COORDINATOR
// ============================================================================

/// Coordinator for federated learning across multiple nodes
///
/// The coordinator manages:
/// - Node registration and heartbeat monitoring
/// - Gradient collection and aggregation
/// - Model update broadcasting
/// - Synchronization rounds
pub struct FederatedCoordinator {
    /// Configuration
    config: FederatedNetworkConfig,

    /// Network backend
    backend: Arc<dyn NetworkBackend>,

    /// Local node information
    local_node: Arc<RwLock<FederatedNode>>,

    /// Known peer nodes
    peers: Arc<RwLock<HashMap<[u8; 32], FederatedNode>>>,

    /// Federated aggregator for gradient processing
    aggregator: Arc<RwLock<FederatedAggregator>>,

    /// Current round number
    current_round: Arc<std::sync::atomic::AtomicU64>,

    /// Statistics
    stats: Arc<RwLock<CoordinatorStats>>,

    /// Event broadcaster
    event_tx: broadcast::Sender<CoordinatorEvent>,

    /// Start time
    start_time: std::time::Instant,

    /// Running flag
    running: Arc<std::sync::atomic::AtomicBool>,

    /// Shutdown signal sender
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl FederatedCoordinator {
    /// Create a new coordinator with a local channel backend (for testing)
    pub async fn new(config: FederatedNetworkConfig, initial_weights: Vec<f32>) -> Self {
        Self::new_with_backend(
            config,
            initial_weights,
            Arc::new(LocalChannelBackend::new()),
        )
        .await
    }

    /// Create a new coordinator with a specific backend
    pub async fn new_with_backend(
        config: FederatedNetworkConfig,
        initial_weights: Vec<f32>,
        backend: Arc<dyn NetworkBackend>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let (shutdown_tx, _shutdown_rx) = oneshot::channel();

        // Create local node
        let local_node = FederatedNode::with_random_id(backend.local_address());

        // Create aggregator with optional DP and Byzantine tolerance
        let mut aggregator = FederatedAggregator::new(initial_weights)
            .with_source_id(local_node.node_id)
            .with_max_staleness(config.max_gradient_staleness_ms);

        if config.enable_dp {
            if let Some(dp_config) = config.dp_config {
                aggregator = aggregator.with_differential_privacy(dp_config);
            }
        }

        if config.enable_byzantine_tolerance {
            aggregator = aggregator.with_byzantine_tolerance(config.byzantine_trim_fraction);
        }

        Self {
            config,
            backend,
            local_node: Arc::new(RwLock::new(local_node)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            aggregator: Arc::new(RwLock::new(aggregator)),
            current_round: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(CoordinatorStats::default())),
            event_tx,
            start_time: std::time::Instant::now(),
            running: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            shutdown_tx: Some(shutdown_tx),
        }
    }

    /// Get the local node ID
    pub fn local_node_id(&self) -> [u8; 32] {
        self.local_node.read().node_id
    }

    /// Get the local node's short ID
    pub fn local_short_id(&self) -> String {
        self.local_node.read().short_id()
    }

    /// Subscribe to coordinator events
    pub fn subscribe_events(&self) -> broadcast::Receiver<CoordinatorEvent> {
        self.event_tx.subscribe()
    }

    /// Get current statistics
    pub fn stats(&self) -> CoordinatorStats {
        let mut stats = self.stats.read().clone();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();
        stats.active_nodes = self.peers.read().len();
        stats.current_round = self.current_round.load(std::sync::atomic::Ordering::SeqCst);
        stats
    }

    /// Get current round number
    pub fn current_round(&self) -> u64 {
        self.current_round.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get number of active peers
    pub fn peer_count(&self) -> usize {
        self.peers.read().len()
    }

    /// Get current model weights
    pub fn get_weights(&self) -> Vec<f32> {
        self.aggregator.read().local_weights().to_vec()
    }

    /// Get a reference to the network backend
    pub fn backend(&self) -> &Arc<dyn NetworkBackend> {
        &self.backend
    }

    /// Update local model weights
    pub fn update_weights(&self, weights: Vec<f32>) {
        self.aggregator.write().update_local_weights(weights);
    }

    /// Register a peer node
    pub async fn register_peer(&self, node: FederatedNode) {
        let node_id = node.node_id;
        let address = node.address.clone();

        // Register with backend for routing (node_id → address mapping)
        if let Err(e) = self.backend.register_node(&node).await {
            tracing::warn!(
                node_id = hex::encode(&node_id[..8]),
                error = %e,
                "Backend node registration failed — peer may lack routing entry"
            );
        }

        self.peers.write().insert(node_id, node);

        info!("Peer registered: {}", hex::encode(&node_id[..8]));

        let _ = self
            .event_tx
            .send(CoordinatorEvent::NodeJoined { node_id, address });
    }

    /// Unregister a peer node
    pub async fn unregister_peer(&self, node_id: &[u8; 32], reason: &str) {
        if self.peers.write().remove(node_id).is_some() {
            info!(
                "Peer unregistered: {} ({})",
                hex::encode(&node_id[..8]),
                reason
            );

            // Also remove from backend routing
            if let Err(e) = self.backend.unregister_node(node_id).await {
                tracing::warn!(
                    "Failed to unregister node {}: {}",
                    hex::encode(&node_id[..8]),
                    e
                );
            }

            let _ = self.event_tx.send(CoordinatorEvent::NodeLeft {
                node_id: *node_id,
                reason: reason.to_string(),
            });
        }
    }

    /// Process an incoming message
    pub async fn process_message(
        &self,
        source: NodeAddress,
        message: FederatedMessage,
    ) -> NetworkResult<()> {
        debug!("Processing {} from {}", message.message_type(), source);

        match message {
            FederatedMessage::GradientShare(gradient) => {
                self.handle_gradient_share(gradient).await?;
            }
            FederatedMessage::Heartbeat {
                node_id,
                round,
                trust_score,
                timestamp,
            } => {
                self.handle_heartbeat(node_id, round, trust_score, timestamp);
            }
            FederatedMessage::JoinRequest {
                node_id,
                address,
                trust_score,
            } => {
                self.handle_join_request(node_id, address, trust_score)
                    .await?;
            }
            FederatedMessage::SyncRequest {
                node_id,
                last_round,
            } => {
                self.handle_sync_request(source, node_id, last_round)
                    .await?;
            }
            FederatedMessage::Leave { node_id, reason } => {
                self.unregister_peer(&node_id, &reason).await;
            }
            FederatedMessage::ModelUpdate { weights, round, .. } => {
                self.handle_model_update(weights, round).await?;
            }
            _ => {
                debug!("Ignoring message type: {}", message.message_type());
            }
        }

        Ok(())
    }

    /// Handle incoming gradient share
    async fn handle_gradient_share(
        &self,
        gradient: crate::swarm::federated_cfc::GradientMessage,
    ) -> NetworkResult<()> {
        debug!(
            "Received gradient from {}",
            hex::encode(&gradient.source_id[..8])
        );

        let accepted = self.aggregator.write().receive_gradient(gradient);

        if accepted {
            self.stats.write().gradients_received += 1;
        }

        Ok(())
    }

    /// Handle heartbeat from a peer
    fn handle_heartbeat(&self, node_id: [u8; 32], round: u64, trust_score: f32, _timestamp: u64) {
        if let Some(peer) = self.peers.write().get_mut(&node_id) {
            peer.update_heartbeat();
            peer.current_round = round;
            peer.trust_score = trust_score;
        }

        self.stats.write().heartbeats_received += 1;
    }

    /// Handle join request
    async fn handle_join_request(
        &self,
        node_id: [u8; 32],
        address: String,
        trust_score: f32,
    ) -> NetworkResult<()> {
        let node = FederatedNode {
            node_id,
            address: NodeAddress::Channel(address.clone()),
            trust_score,
            last_heartbeat: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            current_round: 0,
            is_active: true,
        };

        self.register_peer(node).await;

        // Send acknowledgment (best-effort — the joining peer may not have
        // a backend channel set up yet, e.g. in test scenarios)
        let peers: Vec<([u8; 32], String)> = self
            .peers
            .read()
            .iter()
            .map(|(id, n)| (*id, n.address.to_string()))
            .collect();

        let ack = FederatedMessage::JoinAck {
            node_id,
            current_round: self.current_round(),
            peers,
        };

        if let Err(e) = self.backend.send(&NodeAddress::Channel(address), ack).await {
            debug!("JoinAck delivery failed (peer may connect later): {}", e);
        }

        Ok(())
    }

    /// Handle sync request
    async fn handle_sync_request(
        &self,
        source: NodeAddress,
        _node_id: [u8; 32],
        _last_round: u64,
    ) -> NetworkResult<()> {
        let response = FederatedMessage::SyncResponse {
            weights: self.get_weights(),
            round: self.current_round(),
        };

        self.backend.send(&source, response).await?;

        Ok(())
    }

    /// Handle model update from coordinator
    async fn handle_model_update(&self, weights: Vec<f32>, round: u64) -> NetworkResult<()> {
        if round > self.current_round() {
            self.update_weights(weights);
            self.current_round
                .store(round, std::sync::atomic::Ordering::SeqCst);

            let _ = self.event_tx.send(CoordinatorEvent::ModelUpdated {
                round,
                weights_dim: self.get_weights().len(),
            });
        }

        Ok(())
    }

    /// Share local gradient with all peers
    pub async fn share_gradient(&self, noise_scale: f32) -> NetworkResult<usize> {
        let gradient = self.aggregator.read().export_local_gradient(noise_scale);
        let message = FederatedMessage::GradientShare(gradient);

        let sent = self.backend.broadcast(message).await?;

        debug!("Shared gradient with {} peers", sent);
        Ok(sent)
    }

    /// Send heartbeat to all peers
    pub async fn send_heartbeat(&self) -> NetworkResult<usize> {
        let local_node = self.local_node.read().clone();

        let message = FederatedMessage::Heartbeat {
            node_id: local_node.node_id,
            round: self.current_round(),
            trust_score: local_node.trust_score,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };

        self.backend.broadcast(message).await
    }

    /// Run a single synchronization round
    ///
    /// This collects gradients from peers, aggregates them, and broadcasts
    /// the updated model.
    pub async fn run_sync_round(&self) -> NetworkResult<Option<Vec<f32>>> {
        let round = self
            .current_round
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        info!("Starting sync round {}", round);

        // Check if we have enough peers
        let peer_count = self.peer_count();
        if peer_count < self.config.min_nodes_for_aggregation.saturating_sub(1) {
            warn!(
                "Not enough peers for aggregation ({}/{})",
                peer_count, self.config.min_nodes_for_aggregation
            );
            return Ok(None);
        }

        // Share our gradient
        let noise_scale = if self.config.enable_dp {
            self.config
                .dp_config
                .as_ref()
                .map(|c| c.sigma() as f32)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        self.share_gradient(noise_scale).await?;

        // Wait for gradients from peers (with timeout)
        let timeout = Duration::from_millis(self.config.timeout_ms);
        let deadline = std::time::Instant::now() + timeout;

        while std::time::Instant::now() < deadline {
            // Check if we have enough contributions
            if self.aggregator.read().pending_contributions()
                >= self.config.min_nodes_for_aggregation
            {
                break;
            }

            // Try to receive a message
            match self.backend.receive(Duration::from_millis(100)).await {
                Ok((source, message)) => {
                    if let Err(e) = self.process_message(source, message).await {
                        warn!("Error processing message: {}", e);
                    }
                }
                Err(NetworkError::Timeout { .. }) => {
                    // Continue waiting
                }
                Err(e) => {
                    warn!("Error receiving message: {}", e);
                }
            }
        }

        // Perform aggregation
        let aggregated = self.aggregator.write().aggregate();

        if let Some(ref new_weights) = aggregated {
            // Apply aggregated gradient
            self.aggregator.write().apply_gradient(new_weights, 0.1);

            // Broadcast updated model
            // NOTE: extract node_id and drop the RwLockReadGuard before .await
            // to satisfy Send bound for tokio::spawn.
            let update_msg = {
                let local_node = self.local_node.read();
                FederatedMessage::ModelUpdate {
                    weights: self.get_weights(),
                    round,
                    source_id: local_node.node_id,
                    timestamp: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                }
            }; // guard dropped here

            self.backend.broadcast(update_msg).await?;
            self.stats.write().model_updates_broadcast += 1;

            // Emit event
            let _ = self.event_tx.send(CoordinatorEvent::RoundCompleted {
                round,
                participants: peer_count + 1,
            });

            info!(
                "Round {} completed with {} participants",
                round,
                peer_count + 1
            );
        }

        Ok(aggregated)
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&mut self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);

        // Send leave message to all peers
        let local_node = self.local_node.read().clone();
        let leave_msg = FederatedMessage::Leave {
            node_id: local_node.node_id,
            reason: "Coordinator shutdown".to_string(),
        };

        if let Err(e) = self.backend.broadcast(leave_msg).await {
            tracing::warn!("Failed to broadcast leave message: {}", e);
        }

        // Take and drop the shutdown sender
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }

        info!("Coordinator shutdown complete");
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Create a fully connected local channel network for testing
///
/// Returns a vector of coordinators that can communicate with each other.
pub async fn create_test_network(
    num_nodes: usize,
    weights_dim: usize,
) -> Vec<FederatedCoordinator> {
    let config = FederatedNetworkConfig::for_testing().with_num_nodes(num_nodes);
    let initial_weights = vec![0.0f32; weights_dim];

    // Create backends first
    let backends: Vec<Arc<LocalChannelBackend>> = (0..num_nodes)
        .map(|i| Arc::new(LocalChannelBackend::with_id(format!("node-{i}"))))
        .collect();

    // Cross-register all backends
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                backends[i].register_peer_sender(backends[j].local_id(), backends[j].get_sender());
            }
        }
    }

    // Create coordinators
    let mut coordinators = Vec::new();
    for backend in backends {
        let coordinator = FederatedCoordinator::new_with_backend(
            config.clone(),
            initial_weights.clone(),
            backend,
        )
        .await;
        coordinators.push(coordinator);
    }

    // Register peers with each other
    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                let peer_node = FederatedNode::new(
                    coordinators[j].local_node_id(),
                    NodeAddress::Channel(format!("node-{j}")),
                );
                coordinators[i].register_peer(peer_node).await;
            }
        }
    }

    coordinators
}
