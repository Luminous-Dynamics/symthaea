// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebSocket Subscriptions
//!
//! Real-time updates for emails, notifications, and sync

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::header::AUTHORIZATION,
    response::Response,
};
use futures::{SinkExt, StreamExt};
use jsonwebtoken::{decode, DecodingKey, Validation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum WsMessage {
    // Client -> Server
    Subscribe { channel: String },
    Unsubscribe { channel: String },
    Ping,

    // Server -> Client
    Pong,
    NewEmail(EmailNotification),
    EmailUpdated(EmailUpdateNotification),
    EmailDeleted { id: Uuid },
    TrustScoreUpdated(TrustUpdateNotification),
    SyncStatus(SyncStatusNotification),
    TypingIndicator(TypingNotification),
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailNotification {
    pub id: Uuid,
    pub from: String,
    pub subject: String,
    pub preview: String,
    pub trust_score: f32,
    pub is_encrypted: bool,
    pub received_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailUpdateNotification {
    pub id: Uuid,
    pub changes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustUpdateNotification {
    pub email: String,
    pub old_score: f32,
    pub new_score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStatusNotification {
    pub status: SyncStatus,
    pub progress: Option<f32>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SyncStatus {
    Idle,
    Syncing,
    Error,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypingNotification {
    pub user_id: Uuid,
    pub email_id: Option<Uuid>,
    pub is_typing: bool,
}

/// Connection state for a WebSocket client
struct ClientConnection {
    user_id: Uuid,
    subscriptions: HashSet<String>,
    tx: mpsc::Sender<WsMessage>,
}

use std::collections::HashSet;

/// WebSocket hub managing all connections
pub struct WebSocketHub {
    /// Active connections by connection ID
    connections: RwLock<HashMap<Uuid, ClientConnection>>,
    /// User ID -> connection IDs mapping
    user_connections: RwLock<HashMap<Uuid, HashSet<Uuid>>>,
    /// Broadcast channel for all messages
    broadcast_tx: broadcast::Sender<(String, WsMessage)>,
}

impl WebSocketHub {
    pub fn new() -> Arc<Self> {
        let (broadcast_tx, _) = broadcast::channel(1000);

        Arc::new(Self {
            connections: RwLock::new(HashMap::new()),
            user_connections: RwLock::new(HashMap::new()),
            broadcast_tx,
        })
    }

    /// Handle WebSocket upgrade
    pub async fn handle_upgrade(
        hub: Arc<Self>,
        ws: WebSocketUpgrade,
        user_id: Uuid,
    ) -> Response {
        ws.on_upgrade(move |socket| hub.handle_connection(socket, user_id))
    }

    /// Handle individual WebSocket connection
    async fn handle_connection(self: Arc<Self>, socket: WebSocket, user_id: Uuid) {
        let connection_id = Uuid::new_v4();
        info!(user_id = %user_id, connection_id = %connection_id, "WebSocket connected");

        let (mut ws_tx, mut ws_rx) = socket.split();
        let (tx, mut rx) = mpsc::channel::<WsMessage>(100);

        // Register connection
        {
            let mut connections = self.connections.write().await;
            connections.insert(
                connection_id,
                ClientConnection {
                    user_id,
                    subscriptions: HashSet::new(),
                    tx: tx.clone(),
                },
            );

            let mut user_conns = self.user_connections.write().await;
            user_conns
                .entry(user_id)
                .or_insert_with(HashSet::new)
                .insert(connection_id);
        }

        // Subscribe to broadcast channel
        let mut broadcast_rx = self.broadcast_tx.subscribe();
        let hub = self.clone();

        // Task to send messages to client
        let send_task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Messages from internal queue
                    Some(msg) = rx.recv() => {
                        let text = serde_json::to_string(&msg).unwrap();
                        if ws_tx.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    // Broadcast messages
                    Ok((channel, msg)) = broadcast_rx.recv() => {
                        // Check if this connection is subscribed to the channel
                        let connections = hub.connections.read().await;
                        if let Some(conn) = connections.get(&connection_id) {
                            if conn.subscriptions.contains(&channel) || channel == "all" {
                                let text = serde_json::to_string(&msg).unwrap();
                                if ws_tx.send(Message::Text(text)).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        });

        // Task to receive messages from client
        let hub = self.clone();
        let receive_task = tokio::spawn(async move {
            while let Some(Ok(msg)) = ws_rx.next().await {
                match msg {
                    Message::Text(text) => {
                        if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                            hub.handle_client_message(connection_id, ws_msg).await;
                        }
                    }
                    Message::Ping(data) => {
                        // Pong is handled automatically by axum
                        debug!("Received ping");
                    }
                    Message::Close(_) => {
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Wait for either task to complete
        tokio::select! {
            _ = send_task => {}
            _ = receive_task => {}
        }

        // Cleanup connection
        self.remove_connection(connection_id, user_id).await;
        info!(user_id = %user_id, connection_id = %connection_id, "WebSocket disconnected");
    }

    /// Handle message from client
    async fn handle_client_message(&self, connection_id: Uuid, msg: WsMessage) {
        let mut connections = self.connections.write().await;

        if let Some(conn) = connections.get_mut(&connection_id) {
            match msg {
                WsMessage::Subscribe { channel } => {
                    conn.subscriptions.insert(channel.clone());
                    debug!(connection_id = %connection_id, channel = %channel, "Subscribed");
                }
                WsMessage::Unsubscribe { channel } => {
                    conn.subscriptions.remove(&channel);
                    debug!(connection_id = %connection_id, channel = %channel, "Unsubscribed");
                }
                WsMessage::Ping => {
                    let _ = conn.tx.send(WsMessage::Pong).await;
                }
                _ => {}
            }
        }
    }

    /// Remove connection on disconnect
    async fn remove_connection(&self, connection_id: Uuid, user_id: Uuid) {
        let mut connections = self.connections.write().await;
        connections.remove(&connection_id);

        let mut user_conns = self.user_connections.write().await;
        if let Some(conns) = user_conns.get_mut(&user_id) {
            conns.remove(&connection_id);
            if conns.is_empty() {
                user_conns.remove(&user_id);
            }
        }
    }

    /// Send message to specific user
    pub async fn send_to_user(&self, user_id: Uuid, msg: WsMessage) {
        let connections = self.connections.read().await;
        let user_conns = self.user_connections.read().await;

        if let Some(conn_ids) = user_conns.get(&user_id) {
            for conn_id in conn_ids {
                if let Some(conn) = connections.get(conn_id) {
                    let _ = conn.tx.send(msg.clone()).await;
                }
            }
        }
    }

    /// Broadcast to channel
    pub async fn broadcast(&self, channel: &str, msg: WsMessage) {
        let _ = self.broadcast_tx.send((channel.to_string(), msg));
    }

    /// Broadcast to all connected users
    pub async fn broadcast_all(&self, msg: WsMessage) {
        let _ = self.broadcast_tx.send(("all".to_string(), msg));
    }

    /// Notify about new email
    pub async fn notify_new_email(&self, user_id: Uuid, email: EmailNotification) {
        self.send_to_user(user_id, WsMessage::NewEmail(email)).await;
    }

    /// Notify about trust score change
    pub async fn notify_trust_update(&self, user_id: Uuid, update: TrustUpdateNotification) {
        self.send_to_user(user_id, WsMessage::TrustScoreUpdated(update)).await;
    }

    /// Get connected user count
    pub async fn connected_users_count(&self) -> usize {
        self.user_connections.read().await.len()
    }

    /// Get total connection count
    pub async fn connections_count(&self) -> usize {
        self.connections.read().await.len()
    }
}

impl Default for WebSocketHub {
    fn default() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        Self {
            connections: RwLock::new(HashMap::new()),
            user_connections: RwLock::new(HashMap::new()),
            broadcast_tx,
        }
    }
}

/// JWT claims for WebSocket authentication
#[derive(Debug, Deserialize)]
struct WsClaims {
    /// Subject (user ID)
    sub: String,
    /// Expiration timestamp
    exp: i64,
}

/// Query parameters for WebSocket connection
#[derive(Debug, Deserialize)]
pub struct WsQueryParams {
    /// JWT token passed as query parameter (for WebSocket since headers aren't easily accessible)
    token: Option<String>,
}

/// Configuration for WebSocket authentication
pub struct WsAuthConfig {
    pub jwt_secret: String,
}

impl Default for WsAuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: std::env::var("JWT_SECRET")
                .unwrap_or_else(|_| "mycelix-mail-default-jwt-secret".to_string()),
        }
    }
}

/// Extract user ID from JWT token
fn extract_user_id_from_token(token: &str, secret: &str) -> Option<Uuid> {
    let token = token.strip_prefix("Bearer ").unwrap_or(token);

    let token_data = decode::<WsClaims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .ok()?;

    Uuid::parse_str(&token_data.claims.sub).ok()
}

/// Axum handler for WebSocket connections with authentication
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(hub): State<Arc<WebSocketHub>>,
    Query(params): Query<WsQueryParams>,
) -> Response {
    let auth_config = WsAuthConfig::default();

    // Extract user ID from token in query params
    let user_id = params
        .token
        .as_ref()
        .and_then(|token| extract_user_id_from_token(token, &auth_config.jwt_secret))
        .unwrap_or_else(|| {
            warn!("WebSocket connection without valid auth token, generating anonymous ID");
            Uuid::new_v4() // Fallback for anonymous/unauthenticated connections
        });

    info!(user_id = %user_id, "WebSocket connection authenticated");
    WebSocketHub::handle_upgrade(hub, ws, user_id).await
}

/// Axum handler for WebSocket connections with explicit auth config
pub async fn ws_handler_with_config(
    ws: WebSocketUpgrade,
    State((hub, config)): State<(Arc<WebSocketHub>, Arc<WsAuthConfig>)>,
    Query(params): Query<WsQueryParams>,
) -> Response {
    // Extract user ID from token in query params
    let user_id = params
        .token
        .as_ref()
        .and_then(|token| extract_user_id_from_token(token, &config.jwt_secret))
        .unwrap_or_else(|| {
            warn!("WebSocket connection without valid auth token, generating anonymous ID");
            Uuid::new_v4()
        });

    info!(user_id = %user_id, "WebSocket connection authenticated");
    WebSocketHub::handle_upgrade(hub, ws, user_id).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hub_creation() {
        let hub = WebSocketHub::new();
        assert_eq!(hub.connections_count().await, 0);
        assert_eq!(hub.connected_users_count().await, 0);
    }

    #[test]
    fn test_message_serialization() {
        let msg = WsMessage::NewEmail(EmailNotification {
            id: Uuid::new_v4(),
            from: "sender@example.com".to_string(),
            subject: "Test".to_string(),
            preview: "Preview".to_string(),
            trust_score: 0.8,
            is_encrypted: true,
            received_at: chrono::Utc::now(),
        });

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("NewEmail"));
        assert!(json.contains("sender@example.com"));
    }
}
