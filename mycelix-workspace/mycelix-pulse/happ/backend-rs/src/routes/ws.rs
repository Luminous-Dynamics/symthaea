// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebSocket routes for real-time updates
//!
//! Provides live notifications when:
//! - New mail arrives
//! - Trust scores change
//! - DID resolutions complete

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::routes::AppState;

/// WebSocket event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WsEvent {
    /// New mail received
    NewMail {
        from_did: String,
        subject_preview: String,
        timestamp: i64,
    },
    /// Trust score updated for a DID
    TrustUpdated {
        did: String,
        new_score: f64,
        is_byzantine: bool,
    },
    /// Connection established
    Connected {
        session_id: String,
    },
    /// Heartbeat to keep connection alive
    Ping,
    /// Acknowledgment
    Pong,
}

/// Broadcast channel for WebSocket events
pub type EventBroadcast = broadcast::Sender<WsEvent>;

/// Create WebSocket routes
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/events", get(ws_handler))
}

/// WebSocket connection handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast channel
    let mut rx = state.event_broadcast.subscribe();

    // Generate session ID
    let session_id = uuid::Uuid::new_v4().to_string();

    // Send connected event
    let connected = WsEvent::Connected {
        session_id: session_id.clone(),
    };
    if let Ok(msg) = serde_json::to_string(&connected) {
        let _ = sender.send(Message::Text(msg.into())).await;
    }

    tracing::info!("WebSocket connected: {}", session_id);

    // Spawn task to forward broadcast events to this client
    let forward_task = {
        let session_id = session_id.clone();
        tokio::spawn(async move {
            while let Ok(event) = rx.recv().await {
                match serde_json::to_string(&event) {
                    Ok(msg) => {
                        if sender.send(Message::Text(msg.into())).await.is_err() {
                            tracing::debug!("Client {} disconnected during send", session_id);
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to serialize event: {}", e);
                    }
                }
            }
        })
    };

    // Handle incoming messages from client
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Handle client messages (e.g., subscribe to specific DIDs)
                if let Ok(event) = serde_json::from_str::<ClientMessage>(&text) {
                    match event {
                        ClientMessage::Ping => {
                            // Respond with Pong
                            let _ = state.event_broadcast.send(WsEvent::Pong);
                        }
                        ClientMessage::Subscribe { did } => {
                            tracing::debug!("Client {} subscribed to DID: {}", session_id, did);
                            // In production, track subscriptions per client
                        }
                    }
                }
            }
            Ok(Message::Ping(data)) => {
                // Axum handles Pong automatically
                tracing::trace!("Received ping from {}", session_id);
                let _ = data; // Suppress warning
            }
            Ok(Message::Close(_)) => {
                tracing::info!("Client {} requested close", session_id);
                break;
            }
            Err(e) => {
                tracing::warn!("WebSocket error for {}: {}", session_id, e);
                break;
            }
            _ => {}
        }
    }

    // Cleanup
    forward_task.abort();
    tracing::info!("WebSocket disconnected: {}", session_id);
}

/// Messages from client
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    Ping,
    Subscribe { did: String },
}

/// Helper to broadcast new mail notification
pub fn notify_new_mail(
    broadcast: &EventBroadcast,
    from_did: String,
    subject_preview: String,
    timestamp: i64,
) {
    let event = WsEvent::NewMail {
        from_did,
        subject_preview,
        timestamp,
    };
    let _ = broadcast.send(event);
}

/// Helper to broadcast trust update
pub fn notify_trust_update(
    broadcast: &EventBroadcast,
    did: String,
    new_score: f64,
    is_byzantine: bool,
) {
    let event = WsEvent::TrustUpdated {
        did,
        new_score,
        is_byzantine,
    };
    let _ = broadcast.send(event);
}
