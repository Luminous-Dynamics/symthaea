// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! TCP-based backend for real network communication.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::types::{
    FederatedMessage, FederatedNode, NetworkBackend, NetworkError, NetworkResult, NodeAddress,
};

/// Maximum wire message size: 10 MB
const TCP_MAX_MESSAGE_SIZE: usize = 10_000_000;

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

/// Write a length-prefixed bincode frame to a TCP write half.
///
/// Frame layout: `[4-byte big-endian length][bincode payload]`
///
/// Returns `NetworkError::Serialization` if bincode encoding fails,
/// `NetworkError::SendFailed` if the payload exceeds `TCP_MAX_MESSAGE_SIZE`
/// or the write itself fails.
async fn write_framed(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    msg: &FederatedMessage,
) -> NetworkResult<()> {
    use tokio::io::AsyncWriteExt;

    let data = bincode::serialize(msg).map_err(|e| NetworkError::Serialization {
        reason: e.to_string(),
    })?;

    if data.len() > TCP_MAX_MESSAGE_SIZE {
        return Err(NetworkError::SendFailed {
            reason: format!(
                "Message too large ({} bytes, max {} bytes)",
                data.len(),
                TCP_MAX_MESSAGE_SIZE
            ),
        });
    }

    let len_bytes = (data.len() as u32).to_be_bytes();

    writer
        .write_all(&len_bytes)
        .await
        .map_err(|e| NetworkError::SendFailed {
            reason: e.to_string(),
        })?;

    writer
        .write_all(&data)
        .await
        .map_err(|e| NetworkError::SendFailed {
            reason: e.to_string(),
        })?;

    Ok(())
}

/// Read a length-prefixed bincode frame from a TCP read half.
///
/// Returns `NetworkError::ReceiveFailed` on I/O errors or if the advertised
/// length exceeds `TCP_MAX_MESSAGE_SIZE`, and `NetworkError::Serialization`
/// if bincode decoding fails.
async fn read_framed(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
) -> Result<FederatedMessage, NetworkError> {
    use tokio::io::AsyncReadExt;

    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| NetworkError::ReceiveFailed {
            reason: e.to_string(),
        })?;

    let len = u32::from_be_bytes(len_buf) as usize;

    if len > TCP_MAX_MESSAGE_SIZE {
        return Err(NetworkError::ReceiveFailed {
            reason: format!(
                "Incoming message too large ({len} bytes, max {TCP_MAX_MESSAGE_SIZE} bytes)"
            ),
        });
    }

    let mut data = vec![0u8; len];
    reader
        .read_exact(&mut data)
        .await
        .map_err(|e| NetworkError::ReceiveFailed {
            reason: e.to_string(),
        })?;

    bincode::deserialize(&data).map_err(|e| NetworkError::Serialization {
        reason: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// TcpBackend
// ---------------------------------------------------------------------------

/// TCP-based backend for real network communication
///
/// Provides a fully functional TCP networking layer for the federated swarm.
/// Messages are framed using a 4-byte big-endian length prefix followed by
/// bincode-serialized payload:
///
/// ```text
/// ┌─────────────────────┬──────────────────────────────────┐
/// │ 4 bytes (BE u32)    │  bincode(FederatedMessage)       │
/// │ payload length      │  variable length payload         │
/// └─────────────────────┴──────────────────────────────────┘
/// ```
///
/// # Connection Management
///
/// - Incoming connections are accepted by a background listener task.
/// - Outgoing connections are created lazily on first send and cached.
/// - Each connection is split into an `OwnedReadHalf` (spawned as a reader task)
///   and an `OwnedWriteHalf` (stored for sending).
/// - Connection failures automatically evict stale entries from the cache.
pub struct TcpBackend {
    /// Actual bound local address (resolved from port 0 if needed)
    local_addr: SocketAddr,

    /// Active outgoing TCP connections to peers, keyed by remote address.
    connections:
        Arc<RwLock<HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>>>,

    /// Sender side of the incoming message channel.
    incoming_tx: mpsc::Sender<(NodeAddress, FederatedMessage)>,

    /// Receiver side of the incoming message channel.
    incoming_rx: tokio::sync::Mutex<mpsc::Receiver<(NodeAddress, FederatedMessage)>>,

    /// Mapping from 32-byte node IDs to their socket addresses.
    pub(crate) registered_nodes: Arc<RwLock<HashMap<[u8; 32], SocketAddr>>>,

    /// Handle for the TCP listener accept-loop task.
    #[allow(dead_code)]
    listener_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,

    /// Atomic flag indicating whether the backend has bound and is ready.
    ready: Arc<std::sync::atomic::AtomicBool>,
}

impl TcpBackend {
    /// Create a new TCP backend bound to `bind_addr`.
    ///
    /// This is an async constructor because it:
    /// 1. Binds a `TcpListener` (supports port 0 for OS-assigned ports).
    /// 2. Stores the actual bound address (important when port 0 is used).
    /// 3. Spawns a background accept loop that handles incoming connections.
    ///
    /// # Errors
    ///
    /// Returns `NetworkError::Internal` if the bind fails.
    pub async fn new(bind_addr: SocketAddr) -> NetworkResult<Self> {
        let listener = tokio::net::TcpListener::bind(bind_addr)
            .await
            .map_err(|e| NetworkError::Internal {
                reason: format!("Failed to bind TCP listener on {bind_addr}: {e}"),
            })?;

        let local_addr = listener.local_addr().map_err(|e| NetworkError::Internal {
            reason: format!("Failed to get local address: {e}"),
        })?;

        let (incoming_tx, incoming_rx) = mpsc::channel(4096);

        let connections: Arc<
            RwLock<HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>>,
        > = Arc::new(RwLock::new(HashMap::new()));

        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));

        // Spawn the accept loop
        let accept_tx = incoming_tx.clone();
        let accept_connections = connections.clone();
        let accept_ready = ready.clone();

        let listener_handle = tokio::spawn(async move {
            // Mark ready once the listener is running
            accept_ready.store(true, std::sync::atomic::Ordering::SeqCst);

            loop {
                match listener.accept().await {
                    Ok((stream, peer_addr)) => {
                        debug!("TCP: accepted connection from {}", peer_addr);

                        let (read_half, write_half) = stream.into_split();

                        // Cache the write half so we can send back to this peer
                        accept_connections
                            .write()
                            .insert(peer_addr, Arc::new(tokio::sync::Mutex::new(write_half)));

                        // Spawn a reader task for this connection
                        let reader_tx = accept_tx.clone();
                        let reader_connections = accept_connections.clone();
                        tokio::spawn(Self::reader_loop(
                            read_half,
                            peer_addr,
                            reader_tx,
                            reader_connections,
                        ));
                    }
                    Err(e) => {
                        warn!("TCP: accept error: {}", e);
                        // Brief sleep to avoid a tight error loop
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                }
            }
        });

        Ok(Self {
            local_addr,
            connections,
            incoming_tx,
            incoming_rx: tokio::sync::Mutex::new(incoming_rx),
            registered_nodes: Arc::new(RwLock::new(HashMap::new())),
            listener_handle: tokio::sync::Mutex::new(Some(listener_handle)),
            ready,
        })
    }

    /// Background reader loop for a single TCP stream.
    async fn reader_loop(
        mut read_half: tokio::net::tcp::OwnedReadHalf,
        peer_addr: SocketAddr,
        incoming_tx: mpsc::Sender<(NodeAddress, FederatedMessage)>,
        connections: Arc<
            RwLock<HashMap<SocketAddr, Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>>>,
        >,
    ) {
        loop {
            match read_framed(&mut read_half).await {
                Ok(msg) => {
                    debug!("TCP: received {} from {}", msg.message_type(), peer_addr);
                    let source = NodeAddress::Socket(peer_addr);
                    if incoming_tx.send((source, msg)).await.is_err() {
                        debug!(
                            "TCP: incoming channel closed, stopping reader for {}",
                            peer_addr
                        );
                        break;
                    }
                }
                Err(e) => {
                    debug!("TCP: reader for {} ended: {}", peer_addr, e);
                    connections.write().remove(&peer_addr);
                    break;
                }
            }
        }
    }

    /// Get or create an outgoing TCP connection to `addr`.
    async fn get_or_connect(
        &self,
        addr: SocketAddr,
    ) -> NetworkResult<Arc<tokio::sync::Mutex<tokio::net::tcp::OwnedWriteHalf>>> {
        // Fast path: connection already cached
        {
            let conns = self.connections.read();
            if let Some(writer) = conns.get(&addr) {
                return Ok(Arc::clone(writer));
            }
        }

        // Slow path: establish a new connection
        debug!("TCP: connecting to {}", addr);
        let stream = tokio::net::TcpStream::connect(addr).await.map_err(|e| {
            NetworkError::ConnectionFailed {
                target: addr.to_string(),
                reason: e.to_string(),
            }
        })?;

        let (read_half, write_half) = stream.into_split();
        let writer = Arc::new(tokio::sync::Mutex::new(write_half));

        // Cache the write half
        self.connections.write().insert(addr, Arc::clone(&writer));

        // Spawn a reader task for replies on this connection
        let reader_tx = self.incoming_tx.clone();
        let reader_connections = self.connections.clone();
        tokio::spawn(Self::reader_loop(
            read_half,
            addr,
            reader_tx,
            reader_connections,
        ));

        Ok(writer)
    }

    /// Resolve a `NodeAddress` to a `SocketAddr`.
    fn resolve_addr(target: &NodeAddress) -> NetworkResult<SocketAddr> {
        match target {
            NodeAddress::Socket(addr) => Ok(*addr),
            other => Err(NetworkError::SendFailed {
                reason: format!("TcpBackend requires Socket addresses, got {other}"),
            }),
        }
    }
}

#[async_trait]
impl NetworkBackend for TcpBackend {
    async fn send(&self, target: &NodeAddress, message: FederatedMessage) -> NetworkResult<()> {
        let addr = Self::resolve_addr(target)?;

        let writer = self.get_or_connect(addr).await?;

        let result = {
            let mut guard = writer.lock().await;
            write_framed(&mut guard, &message).await
        };

        if result.is_err() {
            // Evict the broken connection so a future send will reconnect
            self.connections.write().remove(&addr);
        }

        result
    }

    async fn broadcast(&self, message: FederatedMessage) -> NetworkResult<usize> {
        let targets: Vec<([u8; 32], SocketAddr)> = self
            .registered_nodes
            .read()
            .iter()
            .map(|(id, addr)| (*id, *addr))
            .collect();

        let mut success_count = 0usize;

        for (node_id, addr) in &targets {
            let target = NodeAddress::Socket(*addr);
            match self.send(&target, message.clone()).await {
                Ok(()) => {
                    success_count += 1;
                }
                Err(e) => {
                    debug!(
                        "TCP: broadcast to {} ({}) failed: {}",
                        hex::encode(&node_id[..8]),
                        addr,
                        e
                    );
                }
            }
        }

        Ok(success_count)
    }

    async fn receive(&self, timeout: Duration) -> NetworkResult<(NodeAddress, FederatedMessage)> {
        let mut rx = self.incoming_rx.lock().await;

        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(pair)) => Ok(pair),
            Ok(None) => Err(NetworkError::ChannelClosed {
                reason: "Incoming message channel closed".to_string(),
            }),
            Err(_) => Err(NetworkError::Timeout {
                operation: "tcp_receive".to_string(),
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    async fn register_node(&self, node: &FederatedNode) -> NetworkResult<()> {
        let addr = Self::resolve_addr(&node.address)?;
        self.registered_nodes.write().insert(node.node_id, addr);
        debug!(
            "TCP: registered node {} at {}",
            hex::encode(&node.node_id[..8]),
            addr
        );
        Ok(())
    }

    async fn unregister_node(&self, node_id: &[u8; 32]) -> NetworkResult<()> {
        if let Some(addr) = self.registered_nodes.write().remove(node_id) {
            self.connections.write().remove(&addr);
            debug!(
                "TCP: unregistered node {} (was at {})",
                hex::encode(&node_id[..8]),
                addr
            );
        }
        Ok(())
    }

    fn local_address(&self) -> NodeAddress {
        NodeAddress::Socket(self.local_addr)
    }

    fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }
}
