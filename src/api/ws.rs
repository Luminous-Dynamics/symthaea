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
    pub mood_temperature: f32,
    pub thermodynamic_load: f32,
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
    /// 32-dim projection of the thought hypervector for fractal mapping
    pub thought_vector: Vec<f32>,

    // ── Phase 6: Neuromodulator bath telemetry ──
    /// 9-dimensional neuromodulator state vector [DA, NE, 5-HT, ACh, GABA, Oxy, Glut, Aden, ECB]
    #[serde(default)]
    pub neuromod_state_vector: Vec<f32>,
    /// Bath phase space entropy
    #[serde(default)]
    pub bath_entropy: f32,
    /// Allostatic load
    #[serde(default)]
    pub allostatic_load: f32,
    /// E/I ratio (glutamate/GABA)
    #[serde(default)]
    pub ei_ratio: f32,
    /// Sleep pressure (adenosine effective)
    #[serde(default)]
    pub sleep_pressure: f32,
    /// Active injection count
    #[serde(default)]
    pub active_injection_count: u8,
    /// Whether attractor detected in bath phase space
    #[serde(default)]
    pub attractor_detected: bool,

    // ── Mesh Network Telemetry ──
    #[serde(default)]
    pub mesh_health_score: f32,
    #[serde(default)]
    pub mesh_peer_count: u32,
    #[serde(default)]
    pub mesh_bytes_sent: u64,
    #[serde(default)]
    pub mesh_bytes_received: u64,
    #[serde(default)]
    pub mesh_compression_ratio: f64,
    #[serde(default)]
    pub mesh_bandwidth_budget: u64,
    #[serde(default)]
    pub mesh_packets_throttled: u64,

    // ── Post-Phase 6: Phase tracker visualization ──
    /// Bath centroid (9D mean of recent state vectors)
    #[serde(default)]
    pub bath_centroid: Vec<f32>,
    /// Bath per-dimension variance (9D)
    #[serde(default)]
    pub bath_variance: Vec<f32>,
    /// Bath trajectory (last N state vectors)
    #[serde(default)]
    pub bath_trajectory: Vec<Vec<f32>>,
    /// 2D projection: [DA+NE mean, 5-HT+GABA mean]
    #[serde(default)]
    pub bath_projection_2d: Vec<f32>,
    /// Human-readable phase label (stressed/flow/drowsy/alert/relaxed/recovering/balanced)
    #[serde(default)]
    pub bath_phase_label: String,
}

/// Client message format.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ClientMessage {
    TextInput { text: String },
    Thermodynamics { load: f32 },
    Command { command: String },
}

/// WebSocket upgrade handler.
pub async fn ws_handler(ws: WebSocketUpgrade, runner: Arc<Mutex<DemoRunner>>) -> impl IntoResponse {
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
                                ClientMessage::Thermodynamics { load } => {
                                    let mut r = runner.lock().await;
                                    r.update_thermodynamics(load);
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
