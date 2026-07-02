// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! WebSocket handlers for real-time communication

use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::AppState;

/// WebSocket upgrade handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    // Create channel for sending messages to client
    let (tx, mut rx) = mpsc::channel::<WsMessage>(32);

    // Connection ID
    let connection_id = Uuid::new_v4();
    tracing::info!("WebSocket connected: {}", connection_id);

    // Spawn task to forward messages to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let json = serde_json::to_string(&msg).unwrap_or_default();
            if sender.send(Message::Text(json)).await.is_err() {
                break;
            }
        }
    });

    // Send welcome message
    let _ = tx
        .send(WsMessage::Connected {
            connection_id: connection_id.to_string(),
        })
        .await;

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                if let Ok(cmd) = serde_json::from_str::<WsCommand>(&text) {
                    handle_command(cmd, &state, &tx).await;
                }
            }
            Ok(Message::Binary(data)) => {
                // Handle binary audio data
                tracing::debug!("Received {} bytes of binary data", data.len());
            }
            Ok(Message::Ping(data)) => {
                let _ = tx.send(WsMessage::Pong { data }).await;
            }
            Ok(Message::Close(_)) => {
                tracing::info!("WebSocket closed: {}", connection_id);
                break;
            }
            Err(e) => {
                tracing::error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    send_task.abort();
    tracing::info!("WebSocket disconnected: {}", connection_id);
}

/// Handle incoming WebSocket commands
async fn handle_command(cmd: WsCommand, state: &AppState, tx: &mpsc::Sender<WsMessage>) {
    match cmd {
        WsCommand::CreateSession => {
            match state.engine.create_session().await {
                Ok(session_id) => {
                    let _ = tx
                        .send(WsMessage::SessionCreated {
                            session_id: session_id.to_string(),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(WsMessage::Error {
                            message: e.to_string(),
                        })
                        .await;
                }
            }
        }

        WsCommand::JoinSession { session_id } => {
            match Uuid::parse_str(&session_id) {
                Ok(uuid) => match state.engine.get_session(uuid) {
                    Ok(session) => {
                        let session_state = session.state();
                        let _ = tx
                            .send(WsMessage::SessionJoined {
                                session_id,
                                state: SessionStateMsg {
                                    current_track: session_state.current_track.map(|id| id.to_string()),
                                    position: session_state.position,
                                    playback: format!("{:?}", session_state.playback),
                                    volume: session_state.volume,
                                },
                            })
                            .await;
                    }
                    Err(e) => {
                        let _ = tx
                            .send(WsMessage::Error {
                                message: e.to_string(),
                            })
                            .await;
                    }
                },
                Err(_) => {
                    let _ = tx
                        .send(WsMessage::Error {
                            message: "Invalid session ID".to_string(),
                        })
                        .await;
                }
            }
        }

        WsCommand::Play { session_id } => {
            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                if let Ok(session) = state.engine.get_session(uuid) {
                    session.play();
                    let _ = tx
                        .send(WsMessage::PlaybackStateChanged {
                            session_id,
                            state: "Playing".to_string(),
                        })
                        .await;
                }
            }
        }

        WsCommand::Pause { session_id } => {
            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                if let Ok(session) = state.engine.get_session(uuid) {
                    session.pause();
                    let _ = tx
                        .send(WsMessage::PlaybackStateChanged {
                            session_id,
                            state: "Paused".to_string(),
                        })
                        .await;
                }
            }
        }

        WsCommand::Seek { session_id, position } => {
            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                if let Ok(session) = state.engine.get_session(uuid) {
                    session.seek(position);
                    let _ = tx
                        .send(WsMessage::PositionChanged {
                            session_id,
                            position,
                        })
                        .await;
                }
            }
        }

        WsCommand::SetVolume { session_id, volume } => {
            if let Ok(uuid) = Uuid::parse_str(&session_id) {
                if let Ok(session) = state.engine.get_session(uuid) {
                    session.set_volume(volume);
                    let _ = tx
                        .send(WsMessage::VolumeChanged {
                            session_id,
                            volume,
                        })
                        .await;
                }
            }
        }

        WsCommand::SubscribeToSession { session_id } => {
            // In production, would set up event streaming for this session
            let _ = tx
                .send(WsMessage::Subscribed {
                    session_id,
                    events: vec![
                        "playback_state".to_string(),
                        "position".to_string(),
                        "track_changed".to_string(),
                        "peer_joined".to_string(),
                        "peer_left".to_string(),
                    ],
                })
                .await;
        }

        WsCommand::Ping => {
            let _ = tx.send(WsMessage::Pong { data: vec![] }).await;
        }
    }
}

/// Incoming WebSocket commands
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsCommand {
    CreateSession,
    JoinSession { session_id: String },
    Play { session_id: String },
    Pause { session_id: String },
    Seek { session_id: String, position: f32 },
    SetVolume { session_id: String, volume: f32 },
    SubscribeToSession { session_id: String },
    Ping,
}

/// Outgoing WebSocket messages
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    Connected {
        connection_id: String,
    },
    SessionCreated {
        session_id: String,
    },
    SessionJoined {
        session_id: String,
        state: SessionStateMsg,
    },
    PlaybackStateChanged {
        session_id: String,
        state: String,
    },
    PositionChanged {
        session_id: String,
        position: f32,
    },
    VolumeChanged {
        session_id: String,
        volume: f32,
    },
    TrackChanged {
        session_id: String,
        track_id: String,
    },
    PeerJoined {
        session_id: String,
        peer_id: String,
    },
    PeerLeft {
        session_id: String,
        peer_id: String,
    },
    Subscribed {
        session_id: String,
        events: Vec<String>,
    },
    AudioData {
        session_id: String,
        timestamp: f64,
        #[serde(with = "base64_serde")]
        data: Vec<u8>,
    },
    Error {
        message: String,
    },
    Pong {
        data: Vec<u8>,
    },
}

#[derive(Debug, Serialize)]
pub struct SessionStateMsg {
    pub current_track: Option<String>,
    pub position: f32,
    pub playback: String,
    pub volume: f32,
}

/// Base64 serialization for binary data
mod base64_serde {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(bytes: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        STANDARD.decode(&s).map_err(serde::de::Error::custom)
    }
}
