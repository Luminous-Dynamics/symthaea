// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local channel-based backend for testing and simulation.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::debug;

use super::types::{
    FederatedMessage, FederatedNode, NetworkBackend, NetworkError, NetworkResult, NodeAddress,
};

// ============================================================================
// LOCAL CHANNEL BACKEND
// ============================================================================

/// Message wrapper with source address for channel routing
pub struct ChannelEnvelope {
    pub(crate) source: NodeAddress,
    pub(crate) message: FederatedMessage,
}

/// Local channel-based backend for testing and simulation
///
/// This backend uses Tokio MPSC channels to simulate network communication
/// between nodes in the same process. It's ideal for:
/// - Unit and integration tests
/// - Development and debugging
/// - Performance benchmarks without network overhead
pub struct LocalChannelBackend {
    /// Local node's identifier
    local_id: String,

    /// Senders for each registered node
    senders: Arc<RwLock<HashMap<String, mpsc::Sender<ChannelEnvelope>>>>,

    /// Receiver for incoming messages
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<ChannelEnvelope>>>,

    /// Our own sender (for others to send to us)
    self_sender: mpsc::Sender<ChannelEnvelope>,

    /// Mapping from 32-byte node IDs to sender keys (channel IDs).
    /// Populated via `register_node()` and used by `unregister_node()`.
    node_id_map: Arc<RwLock<HashMap<[u8; 32], String>>>,

    /// Whether the backend is initialized
    ready: Arc<std::sync::atomic::AtomicBool>,
}

impl LocalChannelBackend {
    /// Create a new local channel backend
    pub fn new() -> Self {
        Self::with_id(format!("node-{}", rand::random::<u32>()))
    }

    /// Create a new local channel backend with a specific ID
    pub fn with_id(id: String) -> Self {
        let (tx, rx) = mpsc::channel(1000);

        Self {
            local_id: id,
            senders: Arc::new(RwLock::new(HashMap::new())),
            receiver: Arc::new(tokio::sync::Mutex::new(rx)),
            self_sender: tx,
            node_id_map: Arc::new(RwLock::new(HashMap::new())),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    /// Get a sender for this backend (for other nodes to register)
    pub fn get_sender(&self) -> mpsc::Sender<ChannelEnvelope> {
        self.self_sender.clone()
    }

    /// Register another node's sender with this backend
    pub fn register_peer_sender(&self, node_id: &str, sender: mpsc::Sender<ChannelEnvelope>) {
        self.senders.write().insert(node_id.to_string(), sender);
    }

    /// Get the local ID
    pub fn local_id(&self) -> &str {
        &self.local_id
    }
}

impl Default for LocalChannelBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl NetworkBackend for LocalChannelBackend {
    async fn send(&self, target: &NodeAddress, message: FederatedMessage) -> NetworkResult<()> {
        let target_id = match target {
            NodeAddress::Channel(id) => id.clone(),
            _ => {
                return Err(NetworkError::SendFailed {
                    reason: "LocalChannelBackend only supports Channel addresses".to_string(),
                });
            }
        };

        let sender = self.senders.read().get(&target_id).cloned();

        match sender {
            Some(tx) => {
                let envelope = ChannelEnvelope {
                    source: NodeAddress::Channel(self.local_id.clone()),
                    message,
                };

                tx.send(envelope)
                    .await
                    .map_err(|_| NetworkError::ChannelClosed {
                        reason: format!("Channel to {target_id} closed"),
                    })?;

                Ok(())
            }
            None => Err(NetworkError::NodeNotFound { node_id: target_id }),
        }
    }

    async fn broadcast(&self, message: FederatedMessage) -> NetworkResult<usize> {
        let senders: Vec<(String, mpsc::Sender<ChannelEnvelope>)> = self
            .senders
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut sent_count = 0;

        for (node_id, tx) in senders {
            let envelope = ChannelEnvelope {
                source: NodeAddress::Channel(self.local_id.clone()),
                message: message.clone(),
            };

            if tx.send(envelope).await.is_ok() {
                sent_count += 1;
            } else {
                debug!("Failed to send to {}", node_id);
            }
        }

        Ok(sent_count)
    }

    async fn receive(&self, timeout: Duration) -> NetworkResult<(NodeAddress, FederatedMessage)> {
        let mut rx = self.receiver.lock().await;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(envelope)) => Ok((envelope.source, envelope.message)),
            Ok(None) => Err(NetworkError::ChannelClosed {
                reason: "Receiver channel closed".to_string(),
            }),
            Err(_) => Err(NetworkError::Timeout {
                operation: "receive".to_string(),
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    async fn register_node(&self, node: &FederatedNode) -> NetworkResult<()> {
        if let NodeAddress::Channel(ref id) = node.address {
            self.node_id_map.write().insert(node.node_id, id.clone());
        }
        debug!("Node registered: {}", node.short_id());
        Ok(())
    }

    async fn unregister_node(&self, node_id: &[u8; 32]) -> NetworkResult<()> {
        if let Some(key) = self.node_id_map.write().remove(node_id) {
            self.senders.write().remove(&key);
            debug!(
                "Node {} unregistered (key: {})",
                hex::encode(&node_id[..8]),
                key
            );
        }
        Ok(())
    }

    fn local_address(&self) -> NodeAddress {
        NodeAddress::Channel(self.local_id.clone())
    }

    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}
