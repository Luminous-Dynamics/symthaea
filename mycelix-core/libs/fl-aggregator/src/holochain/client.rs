//! Holochain WebSocket client for conductor communication.
//!
//! Provides async WebSocket-based connection to Holochain conductor
//! for zome function calls and admin operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot, RwLock};
use tracing::{debug, error, info, warn};

use super::types::*;

#[cfg(feature = "holochain")]
use futures_util::{SinkExt, StreamExt};
#[cfg(feature = "holochain")]
use tokio_tungstenite::{connect_async, tungstenite::Message};

/// Configuration for Holochain client.
#[derive(Clone, Debug)]
pub struct HolochainConfig {
    /// Admin interface WebSocket URL.
    pub admin_url: String,
    /// App interface WebSocket URL.
    pub app_url: String,
    /// Connection timeout.
    pub timeout: Duration,
    /// Reconnect interval on disconnect.
    pub reconnect_interval: Duration,
    /// Maximum reconnect attempts.
    pub max_reconnect_attempts: u32,
    /// Cell ID for the FL zome (DNA hash + Agent pubkey).
    pub cell_id: Option<(String, String)>,
    /// Zome names.
    pub gradient_zome: String,
    pub credit_zome: String,
    pub reputation_zome: String,
}

impl Default for HolochainConfig {
    fn default() -> Self {
        Self {
            admin_url: "ws://127.0.0.1:65000".to_string(),
            app_url: "ws://127.0.0.1:65001".to_string(),
            timeout: Duration::from_secs(30),
            reconnect_interval: Duration::from_secs(5),
            max_reconnect_attempts: 10,
            cell_id: None,
            gradient_zome: "gradient_storage".to_string(),
            credit_zome: "zerotrustml_credits".to_string(),
            reputation_zome: "reputation_tracker".to_string(),
        }
    }
}

impl HolochainConfig {
    /// Set admin URL.
    pub fn with_admin_url(mut self, url: impl Into<String>) -> Self {
        self.admin_url = url.into();
        self
    }

    /// Set app URL.
    pub fn with_app_url(mut self, url: impl Into<String>) -> Self {
        self.app_url = url.into();
        self
    }

    /// Set cell ID.
    pub fn with_cell_id(mut self, dna_hash: impl Into<String>, agent: impl Into<String>) -> Self {
        self.cell_id = Some((dna_hash.into(), agent.into()));
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Internal message ID counter.
static MSG_ID: AtomicU64 = AtomicU64::new(1);

fn next_msg_id() -> u64 {
    MSG_ID.fetch_add(1, Ordering::SeqCst)
}

/// Holochain conductor request envelope.
#[derive(Debug, Serialize)]
struct HolochainRequest {
    id: u64,
    #[serde(rename = "type")]
    request_type: String,
    data: serde_json::Value,
}

/// Holochain conductor response envelope.
#[derive(Debug, Deserialize)]
struct HolochainResponse {
    id: u64,
    #[serde(rename = "type")]
    response_type: String,
    data: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
}

/// Connection state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
}

/// Pending request handler.
type PendingRequest = oneshot::Sender<Result<serde_json::Value, HolochainError>>;

/// Holochain WebSocket client.
pub struct HolochainClient {
    config: HolochainConfig,
    state: Arc<RwLock<ConnectionState>>,
    pending_requests: Arc<RwLock<HashMap<u64, PendingRequest>>>,
    #[cfg(feature = "holochain")]
    admin_tx: Arc<RwLock<Option<mpsc::Sender<Message>>>>,
    #[cfg(feature = "holochain")]
    app_tx: Arc<RwLock<Option<mpsc::Sender<Message>>>>,
    signal_tx: Option<mpsc::Sender<HolochainSignal>>,
}

impl HolochainClient {
    /// Create a new Holochain client.
    pub fn new(config: HolochainConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "holochain")]
            admin_tx: Arc::new(RwLock::new(None)),
            #[cfg(feature = "holochain")]
            app_tx: Arc::new(RwLock::new(None)),
            signal_tx: None,
        }
    }

    /// Set signal channel for receiving DHT signals.
    pub fn with_signal_channel(mut self, tx: mpsc::Sender<HolochainSignal>) -> Self {
        self.signal_tx = Some(tx);
        self
    }

    /// Get current connection state.
    pub async fn state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Check if connected.
    pub async fn is_connected(&self) -> bool {
        *self.state.read().await == ConnectionState::Connected
    }

    /// Connect to Holochain conductor.
    #[cfg(feature = "holochain")]
    pub async fn connect(&self) -> HolochainResult<()> {
        use tokio_tungstenite::MaybeTlsStream;
        use tokio::net::TcpStream;

        type WsStream = tokio_tungstenite::WebSocketStream<MaybeTlsStream<TcpStream>>;

        {
            let mut state = self.state.write().await;
            if *state == ConnectionState::Connected {
                return Ok(());
            }
            *state = ConnectionState::Connecting;
        }

        info!(admin_url = %self.config.admin_url, app_url = %self.config.app_url, "Connecting to Holochain conductor");

        // Connect to admin interface
        let admin_url_str = &self.config.admin_url;

        let admin_result: Result<(WsStream, _), _> =
            tokio::time::timeout(self.config.timeout, connect_async(admin_url_str))
                .await
                .map_err(|_| HolochainError::Timeout("Admin connection timeout".to_string()))?;

        let (admin_ws, _) = admin_result
            .map_err(|e| HolochainError::WebSocket(e.to_string()))?;

        let (admin_write, admin_read) = admin_ws.split();
        let (admin_tx, admin_rx) = mpsc::channel::<Message>(32);

        // Spawn admin writer task
        tokio::spawn(Self::writer_task(admin_rx, admin_write));

        // Spawn admin reader task
        let pending = self.pending_requests.clone();
        tokio::spawn(Self::reader_task(admin_read, pending.clone(), None));

        *self.admin_tx.write().await = Some(admin_tx);

        // Connect to app interface
        let app_url_str = &self.config.app_url;

        let app_result: Result<(WsStream, _), _> =
            tokio::time::timeout(self.config.timeout, connect_async(app_url_str))
                .await
                .map_err(|_| HolochainError::Timeout("App connection timeout".to_string()))?;

        let (app_ws, _) = app_result
            .map_err(|e| HolochainError::WebSocket(e.to_string()))?;

        let (app_write, app_read) = app_ws.split();
        let (app_tx, app_rx) = mpsc::channel::<Message>(32);

        // Spawn app writer task
        tokio::spawn(Self::writer_task(app_rx, app_write));

        // Spawn app reader task with signal handler
        let signal_tx = self.signal_tx.clone();
        tokio::spawn(Self::reader_task(app_read, pending, signal_tx));

        *self.app_tx.write().await = Some(app_tx);
        *self.state.write().await = ConnectionState::Connected;

        info!("Connected to Holochain conductor");
        Ok(())
    }

    /// Connect stub for non-holochain builds.
    #[cfg(not(feature = "holochain"))]
    pub async fn connect(&self) -> HolochainResult<()> {
        Err(HolochainError::Connection(
            "Holochain feature not enabled".to_string(),
        ))
    }

    /// Writer task that sends messages from channel to WebSocket.
    #[cfg(feature = "holochain")]
    async fn writer_task(
        mut rx: mpsc::Receiver<Message>,
        mut writer: futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
    ) {
        while let Some(msg) = rx.recv().await {
            if let Err(e) = writer.send(msg).await {
                error!("WebSocket write error: {}", e);
                break;
            }
        }
        debug!("Writer task ended");
    }

    /// Reader task that receives messages and dispatches responses.
    #[cfg(feature = "holochain")]
    async fn reader_task(
        mut reader: futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
        pending: Arc<RwLock<HashMap<u64, PendingRequest>>>,
        signal_tx: Option<mpsc::Sender<HolochainSignal>>,
    ) {
        while let Some(result) = reader.next().await {
            match result {
                Ok(Message::Binary(data)) => {
                    // Decode MessagePack response
                    match rmp_serde::from_slice::<HolochainResponse>(&data) {
                        Ok(response) => {
                            // Check if it's a signal
                            if response.response_type == "signal" {
                                if let (Some(tx), Some(data)) = (&signal_tx, response.data) {
                                    if let Ok(signal) =
                                        serde_json::from_value::<HolochainSignal>(data)
                                    {
                                        let _ = tx.send(signal).await;
                                    }
                                }
                                continue;
                            }

                            // Find and complete pending request
                            let sender = {
                                let mut pending = pending.write().await;
                                pending.remove(&response.id)
                            };

                            if let Some(sender) = sender {
                                let result = if let Some(error) = response.error {
                                    Err(HolochainError::ZomeCall {
                                        zome: "unknown".to_string(),
                                        function: "unknown".to_string(),
                                        message: error.to_string(),
                                    })
                                } else {
                                    Ok(response.data.unwrap_or(serde_json::Value::Null))
                                };
                                let _ = sender.send(result);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to decode response: {}", e);
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    info!("WebSocket closed by server");
                    break;
                }
                Err(e) => {
                    error!("WebSocket read error: {}", e);
                    break;
                }
                _ => {}
            }
        }
        debug!("Reader task ended");
    }

    /// Disconnect from Holochain conductor.
    #[cfg(feature = "holochain")]
    pub async fn disconnect(&self) {
        *self.admin_tx.write().await = None;
        *self.app_tx.write().await = None;
        *self.state.write().await = ConnectionState::Disconnected;
        info!("Disconnected from Holochain conductor");
    }

    #[cfg(not(feature = "holochain"))]
    pub async fn disconnect(&self) {
        *self.state.write().await = ConnectionState::Disconnected;
    }

    /// Call admin interface function.
    #[cfg(feature = "holochain")]
    async fn call_admin(&self, request_type: &str, data: serde_json::Value) -> HolochainResult<serde_json::Value> {
        let tx = self.admin_tx.read().await;
        let tx = tx.as_ref().ok_or(HolochainError::NotConnected)?;

        let msg_id = next_msg_id();
        let request = HolochainRequest {
            id: msg_id,
            request_type: request_type.to_string(),
            data,
        };

        let (response_tx, response_rx) = oneshot::channel();
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(msg_id, response_tx);
        }

        let payload = rmp_serde::to_vec(&request)
            .map_err(|e| HolochainError::Serialization(e.to_string()))?;

        tx.send(Message::Binary(payload.into()))
            .await
            .map_err(|e| HolochainError::WebSocket(e.to_string()))?;

        tokio::time::timeout(self.config.timeout, response_rx)
            .await
            .map_err(|_| HolochainError::Timeout("Admin call timeout".to_string()))?
            .map_err(|_| HolochainError::Connection("Response channel closed".to_string()))?
    }

    /// Call zome function.
    #[cfg(feature = "holochain")]
    pub async fn call_zome(
        &self,
        zome: &str,
        function: &str,
        payload: serde_json::Value,
    ) -> HolochainResult<serde_json::Value> {
        let cell_id = self.config.cell_id.as_ref().ok_or(HolochainError::NoCellId)?;

        let tx = self.app_tx.read().await;
        let tx = tx.as_ref().ok_or(HolochainError::NotConnected)?;

        let msg_id = next_msg_id();
        let request = HolochainRequest {
            id: msg_id,
            request_type: "call_zome".to_string(),
            data: serde_json::json!({
                "cell_id": [cell_id.0, cell_id.1],
                "zome_name": zome,
                "fn_name": function,
                "payload": payload,
                "provenance": cell_id.1,
            }),
        };

        let (response_tx, response_rx) = oneshot::channel();
        {
            let mut pending = self.pending_requests.write().await;
            pending.insert(msg_id, response_tx);
        }

        let msg_payload = rmp_serde::to_vec(&request)
            .map_err(|e| HolochainError::Serialization(e.to_string()))?;

        debug!(zome = %zome, function = %function, "Calling zome function");

        tx.send(Message::Binary(msg_payload.into()))
            .await
            .map_err(|e| HolochainError::WebSocket(e.to_string()))?;

        tokio::time::timeout(self.config.timeout, response_rx)
            .await
            .map_err(|_| HolochainError::Timeout(format!("{}.{} timeout", zome, function)))?
            .map_err(|_| HolochainError::Connection("Response channel closed".to_string()))?
    }

    #[cfg(not(feature = "holochain"))]
    pub async fn call_zome(
        &self,
        _zome: &str,
        _function: &str,
        _payload: serde_json::Value,
    ) -> HolochainResult<serde_json::Value> {
        Err(HolochainError::NotConnected)
    }

    // ==================== Gradient Storage Operations ====================

    /// Store a gradient record in the DHT.
    pub async fn store_gradient(&self, record: &GradientRecord) -> HolochainResult<EntryHash> {
        let result = self
            .call_zome(
                &self.config.gradient_zome,
                "store_gradient",
                serde_json::to_value(record)
                    .map_err(|e| HolochainError::Serialization(e.to_string()))?,
            )
            .await?;

        result
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| HolochainError::Encoding("Invalid entry hash response".to_string()))
    }

    /// Get a gradient record by entry hash.
    pub async fn get_gradient(&self, entry_hash: &str) -> HolochainResult<Option<GradientRecord>> {
        let result = self
            .call_zome(
                &self.config.gradient_zome,
                "get_gradient",
                serde_json::json!({ "entry_hash": entry_hash }),
            )
            .await?;

        if result.is_null() {
            return Ok(None);
        }

        serde_json::from_value(result)
            .map(Some)
            .map_err(|e| HolochainError::Encoding(e.to_string()))
    }

    /// Get all gradients for a specific round.
    pub async fn get_gradients_by_round(&self, round_num: u64) -> HolochainResult<Vec<GradientRecord>> {
        let result = self
            .call_zome(
                &self.config.gradient_zome,
                "get_gradients_by_round",
                serde_json::json!({ "round_num": round_num }),
            )
            .await?;

        serde_json::from_value(result).map_err(|e| HolochainError::Encoding(e.to_string()))
    }

    /// Get gradients for a specific node.
    pub async fn get_gradients_by_node(&self, node_id: &str) -> HolochainResult<Vec<GradientRecord>> {
        let result = self
            .call_zome(
                &self.config.gradient_zome,
                "get_gradients_by_node",
                serde_json::json!({ "node_id": node_id }),
            )
            .await?;

        serde_json::from_value(result).map_err(|e| HolochainError::Encoding(e.to_string()))
    }

    // ==================== Credit System Operations ====================

    /// Issue credits to a node.
    pub async fn issue_credit(
        &self,
        holder: &str,
        amount: u64,
        reason: EarnReason,
    ) -> HolochainResult<EntryHash> {
        let result = self
            .call_zome(
                &self.config.credit_zome,
                "issue_credit",
                serde_json::json!({
                    "holder": holder,
                    "amount": amount,
                    "earned_from": reason,
                }),
            )
            .await?;

        result
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| HolochainError::Encoding("Invalid credit hash response".to_string()))
    }

    /// Get credit balance for a holder.
    pub async fn get_credit_balance(&self, holder: &str) -> HolochainResult<u64> {
        let result = self
            .call_zome(
                &self.config.credit_zome,
                "get_balance",
                serde_json::json!({ "holder": holder }),
            )
            .await?;

        result
            .as_u64()
            .ok_or_else(|| HolochainError::Encoding("Invalid balance response".to_string()))
    }

    /// Get credit history for a holder.
    pub async fn get_credit_history(&self, holder: &str) -> HolochainResult<Vec<CreditRecord>> {
        let result = self
            .call_zome(
                &self.config.credit_zome,
                "get_credit_history",
                serde_json::json!({ "holder": holder }),
            )
            .await?;

        serde_json::from_value(result).map_err(|e| HolochainError::Encoding(e.to_string()))
    }

    // ==================== Reputation Operations ====================

    /// Get reputation score for a node.
    pub async fn get_reputation(&self, node_id: &str) -> HolochainResult<f32> {
        let result = self
            .call_zome(
                &self.config.reputation_zome,
                "get_reputation",
                serde_json::json!({ "node_id": node_id }),
            )
            .await?;

        result
            .as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| HolochainError::Encoding("Invalid reputation response".to_string()))
    }

    /// Update reputation for a node.
    pub async fn update_reputation(
        &self,
        node_id: &str,
        delta: f32,
        reason: &str,
    ) -> HolochainResult<f32> {
        let result = self
            .call_zome(
                &self.config.reputation_zome,
                "update_reputation",
                serde_json::json!({
                    "node_id": node_id,
                    "delta": delta,
                    "reason": reason,
                }),
            )
            .await?;

        result
            .as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| HolochainError::Encoding("Invalid new reputation response".to_string()))
    }

    /// Log a Byzantine event.
    pub async fn log_byzantine_event(&self, event: &ByzantineEvent) -> HolochainResult<EntryHash> {
        let result = self
            .call_zome(
                &self.config.reputation_zome,
                "log_byzantine_event",
                serde_json::to_value(event)
                    .map_err(|e| HolochainError::Serialization(e.to_string()))?,
            )
            .await?;

        result
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| HolochainError::Encoding("Invalid event hash response".to_string()))
    }

    /// Get Byzantine events for a node.
    pub async fn get_byzantine_events(&self, node_id: &str) -> HolochainResult<Vec<ByzantineEvent>> {
        let result = self
            .call_zome(
                &self.config.reputation_zome,
                "get_byzantine_events",
                serde_json::json!({ "node_id": node_id }),
            )
            .await?;

        serde_json::from_value(result).map_err(|e| HolochainError::Encoding(e.to_string()))
    }

    // ==================== Health & Admin ====================

    /// Health check - verify connection and cell availability.
    pub async fn health_check(&self) -> HolochainResult<bool> {
        if !self.is_connected().await {
            return Ok(false);
        }

        // Try a simple zome call
        match self
            .call_zome(&self.config.gradient_zome, "health_check", serde_json::json!({}))
            .await
        {
            Ok(_) => Ok(true),
            Err(HolochainError::Timeout(_)) => Ok(false),
            Err(e) => {
                warn!("Health check failed: {}", e);
                Ok(false)
            }
        }
    }

    /// Get conductor info.
    #[cfg(feature = "holochain")]
    pub async fn get_conductor_info(&self) -> HolochainResult<serde_json::Value> {
        self.call_admin("dump_full_state", serde_json::json!({})).await
    }

    #[cfg(not(feature = "holochain"))]
    pub async fn get_conductor_info(&self) -> HolochainResult<serde_json::Value> {
        Err(HolochainError::NotConnected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = HolochainConfig::default()
            .with_admin_url("ws://localhost:9000")
            .with_app_url("ws://localhost:9001")
            .with_cell_id("dna_hash_abc", "agent_pubkey_xyz")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(config.admin_url, "ws://localhost:9000");
        assert_eq!(config.app_url, "ws://localhost:9001");
        assert_eq!(
            config.cell_id,
            Some(("dna_hash_abc".to_string(), "agent_pubkey_xyz".to_string()))
        );
        assert_eq!(config.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_msg_id_increments() {
        let id1 = next_msg_id();
        let id2 = next_msg_id();
        assert!(id2 > id1);
    }

    #[tokio::test]
    async fn test_client_initial_state() {
        let client = HolochainClient::new(HolochainConfig::default());
        assert_eq!(client.state().await, ConnectionState::Disconnected);
        assert!(!client.is_connected().await);
    }

    #[tokio::test]
    async fn test_client_disconnect() {
        let client = HolochainClient::new(HolochainConfig::default());
        client.disconnect().await;
        assert_eq!(client.state().await, ConnectionState::Disconnected);
    }
}
