// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-Time WebSocket API endpoint

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::events::{ClientMessage, ServerMessage, ErrorMessage};
use super::hub::CollaborationHub;
use crate::auth::jwt::JwtService;

/// WebSocket API state
#[derive(Clone)]
pub struct RealtimeState {
    pub hub: Arc<CollaborationHub>,
    pub jwt_service: Arc<JwtService>,
}

impl RealtimeState {
    pub fn new() -> Self {
        Self {
            hub: Arc::new(CollaborationHub::new()),
            jwt_service: Arc::new(JwtService::from_env()),
        }
    }

    pub fn with_jwt_service(jwt_service: JwtService) -> Self {
        Self {
            hub: Arc::new(CollaborationHub::new()),
            jwt_service: Arc::new(jwt_service),
        }
    }
}

impl Default for RealtimeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Create realtime router
pub fn realtime_router(state: RealtimeState) -> Router {
    Router::new()
        .route("/v1/realtime/ws", get(websocket_handler))
        .route("/v1/realtime/stats", get(get_stats))
        .with_state(state)
}

// ============================================================================
// Request Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct WebSocketQuery {
    pub user_id: Uuid,
    pub user_name: String,
    pub tenant_id: Uuid,
    pub token: Option<String>,  // JWT for authentication
}

// ============================================================================
// Handlers
// ============================================================================

async fn websocket_handler(
    ws: WebSocketUpgrade,
    Query(query): Query<WebSocketQuery>,
    State(state): State<RealtimeState>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Validate JWT token - required for WebSocket connections
    if let Some(ref token) = query.token {
        match state.jwt_service.validate_access_token(token) {
            Ok(claims) => {
                // Verify the token user matches the query parameters
                if claims.sub != query.user_id {
                    return Err((
                        StatusCode::UNAUTHORIZED,
                        Json(json!({
                            "error": "token_user_mismatch",
                            "message": "JWT user does not match connection user_id"
                        })),
                    ));
                }
                // Verify tenant if present in token
                if let Some(token_tenant) = claims.tenant_id {
                    if token_tenant != query.tenant_id {
                        return Err((
                            StatusCode::UNAUTHORIZED,
                            Json(json!({
                                "error": "token_tenant_mismatch",
                                "message": "JWT tenant does not match connection tenant_id"
                            })),
                        ));
                    }
                }
                tracing::info!(
                    user_id = %claims.sub,
                    email = %claims.email,
                    "WebSocket JWT validated successfully"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "WebSocket JWT validation failed");
                return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "error": "invalid_token",
                        "message": format!("JWT validation failed: {}", e)
                    })),
                ));
            }
        }
    } else {
        // Token is required in production
        if std::env::var("ALLOW_ANONYMOUS_WS").unwrap_or_default() != "true" {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "token_required",
                    "message": "JWT token is required for WebSocket connections"
                })),
            ));
        }
        tracing::warn!(
            user_id = %query.user_id,
            "WebSocket connection without JWT (anonymous mode enabled)"
        );
    }

    Ok(ws.on_upgrade(move |socket| handle_socket(socket, state, query)))
}

async fn handle_socket(socket: WebSocket, state: RealtimeState, query: WebSocketQuery) {
    let (mut sender, mut receiver) = socket.split();

    // Create channel for sending messages to this client
    let (tx, mut rx) = mpsc::unbounded_channel::<ServerMessage>();

    // Register session with hub
    let session_id = state
        .hub
        .register_session(query.user_id, query.user_name.clone(), query.tenant_id, tx)
        .await;

    tracing::info!(
        session_id = %session_id,
        user_id = %query.user_id,
        tenant_id = %query.tenant_id,
        "WebSocket connected"
    );

    // Spawn task to forward messages from hub to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            match serde_json::to_string(&msg) {
                Ok(json) => {
                    if sender.send(Message::Text(json)).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to serialize message: {}", e);
                }
            }
        }
    });

    // Handle incoming messages from WebSocket
    let hub = state.hub.clone();
    let recv_session_id = session_id;
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(client_msg) => {
                            if let Err(e) = hub.handle_message(recv_session_id, client_msg).await {
                                tracing::error!("Error handling message: {}", e);
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Invalid message format: {}", e);
                        }
                    }
                }
                Message::Binary(_) => {
                    tracing::warn!("Binary messages not supported");
                }
                Message::Ping(data) => {
                    // Axum handles pong automatically
                    tracing::trace!("Received ping: {:?}", data);
                }
                Message::Pong(_) => {
                    tracing::trace!("Received pong");
                }
                Message::Close(_) => {
                    tracing::info!(session_id = %recv_session_id, "WebSocket close requested");
                    break;
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {
            tracing::debug!(session_id = %session_id, "Send task completed");
        }
        _ = recv_task => {
            tracing::debug!(session_id = %session_id, "Receive task completed");
        }
    }

    // Unregister session
    state.hub.unregister_session(session_id).await;

    tracing::info!(
        session_id = %session_id,
        user_id = %query.user_id,
        "WebSocket disconnected"
    );
}

async fn get_stats(State(state): State<RealtimeState>) -> impl IntoResponse {
    let stats = state.hub.get_stats().await;

    (
        StatusCode::OK,
        Json(json!({
            "total_connections": stats.total_connections,
            "active_users": stats.active_users,
            "active_tenants": stats.active_tenants,
            "total_subscriptions": stats.total_subscriptions,
            "active_documents": stats.active_documents
        })),
    )
}
