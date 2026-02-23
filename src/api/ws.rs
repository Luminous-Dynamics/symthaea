//! WebSocket handler for live consciousness telemetry streaming.
//!
//! Connects a browser client to a running CognitiveLoopService, streaming
//! compact cycle metadata at configurable frequency (default 10Hz).
//!
//! ## Protocol
//!
//! - **Server → Client**: JSON `DemoCycleData` per cycle
//! - **Client → Server**: JSON `{"text": "..."}` to set next cycle input
//! - **Client → Server**: JSON `{"command": "pause"|"resume"|"reset"}` for control

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

use super::demo_runner::DemoRunner;

/// Compact telemetry payload streamed per cycle.
///
/// A small subset of CycleMetadata fields chosen for visualization.
#[derive(Debug, Clone, Serialize)]
pub struct DemoCycleData {
    pub cycle: usize,
    pub prediction_error: f32,
    pub consciousness_level: f64,
    pub narrative_self_psi: f64,
    pub valence: f32,
    pub arousal: f32,
    pub moral_score: f64,
    pub coherence: f64,
    pub flow_state: bool,
    pub cycle_time_us: u64,
    pub surprise_triggered: bool,
    pub gwt_broadcast: bool,
    pub dream_insights: usize,
    pub reasoning_confidence: f32,
    pub resonance_frequency: f64,
    pub input_text: String,
}

/// Client message format.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClientMessage {
    TextInput { text: String },
    Command { command: String },
}

/// WebSocket upgrade handler.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    runner: Arc<Mutex<DemoRunner>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, runner))
}

async fn handle_socket(mut socket: WebSocket, runner: Arc<Mutex<DemoRunner>>) {
    // Send initial hello
    let hello = serde_json::json!({
        "type": "connected",
        "message": "Symthaea consciousness telemetry stream",
        "version": env!("CARGO_PKG_VERSION"),
    });
    if socket
        .send(Message::Text(hello.to_string().into()))
        .await
        .is_err()
    {
        return;
    }

    let mut interval = tokio::time::interval(std::time::Duration::from_millis(100)); // 10Hz
    let mut paused = false;

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if paused {
                    continue;
                }

                let data = {
                    let mut r = runner.lock().await;
                    r.run_cycle()
                };

                let json = match serde_json::to_string(&data) {
                    Ok(j) => j,
                    Err(_) => continue,
                };

                if socket.send(Message::Text(json.into())).await.is_err() {
                    break; // Client disconnected
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(client_msg) = serde_json::from_str::<ClientMessage>(&text) {
                            match client_msg {
                                ClientMessage::TextInput { text } => {
                                    let mut r = runner.lock().await;
                                    r.set_input(&text);
                                }
                                ClientMessage::Command { command } => {
                                    match command.as_str() {
                                        "pause" => paused = true,
                                        "resume" => paused = false,
                                        "reset" => {
                                            let mut r = runner.lock().await;
                                            r.reset();
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
}
