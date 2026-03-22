//! Holon REST API: Desktop-side HTTP endpoints for Soma mobile bridge.
//!
//! Implements the `/holon/*` endpoints that `HolonWebSocket.kt` (Android) expects:
//! - `GET  /holon/status`        — Health check + capability discovery
//! - `POST /holon/outbound`      — Phone→Desktop consciousness messages
//! - `GET  /holon/inbound`       — Desktop→Phone queued responses
//! - `POST /holon/consciousness` — Bidirectional consciousness state exchange
//! - `POST /holon/broca`         — Desktop Broca text generation (stub)
//! - `POST /holon/converse`      — Full conversation
//! - `POST /holon/tts`           — TTS synthesis (stub)
//!
//! The router accepts `Arc<RwLock<Symthaea>>` as shared state — the daemon
//! must wrap its Symthaea instance and pass the same Arc to both the
//! consciousness loop and this router.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::consciousness::holon_receiver::SomaMessage;
use crate::Symthaea;

/// Shared state type for the Holon router.
pub type HolonSharedState = Arc<RwLock<Symthaea>>;

/// Build the Holon REST API router.
pub fn holon_router(state: HolonSharedState) -> Router {
    Router::new()
        .route("/holon/status", get(holon_status))
        .route("/holon/outbound", post(holon_outbound))
        .route("/holon/inbound", get(holon_inbound))
        .route("/holon/consciousness", post(holon_consciousness))
        .route("/holon/broca", post(holon_broca))
        .route("/holon/converse", post(holon_converse))
        .route("/holon/tts", post(holon_tts))
        .with_state(state)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Status
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct StatusResponse {
    status: String,
    consciousness: f32,
    has_tts: bool,
    has_broca: bool,
}

async fn holon_status(State(state): State<HolonSharedState>) -> Json<StatusResponse> {
    let s = state.read().await;
    let consciousness = s.introspect().consciousness_level;
    Json(StatusResponse {
        status: "ok".to_string(),
        consciousness,
        has_tts: cfg!(feature = "voice-tts"),
        has_broca: cfg!(feature = "ssm_language"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Outbound (Phone → Desktop)
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_outbound(State(state): State<HolonSharedState>, body: String) -> StatusCode {
    // Parse outside lock
    let messages: Vec<serde_json::Value> = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => vec![v],
            Err(_) => return StatusCode::BAD_REQUEST,
        },
    };

    let device_id = "soma-phone".to_string();

    // Lock briefly to enqueue
    {
        let mut s = state.write().await;
        for msg_value in messages {
            if let Ok(msg) = serde_json::from_value::<SomaMessage>(msg_value) {
                s.holon_enqueue_soma_message(device_id.clone(), msg);
            }
        }
        // Process immediately so responses are available on next inbound poll
        s.holon_process_pending();
    }

    StatusCode::OK
}

// ═══════════════════════════════════════════════════════════════════════════════
// Inbound (Desktop → Phone)
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_inbound(State(state): State<HolonSharedState>) -> impl IntoResponse {
    let mut s = state.write().await;
    let responses = s.holon_drain_soma_outbound("soma-phone");
    let json = serde_json::to_string(&responses).unwrap_or_else(|_| "[]".to_string());
    (StatusCode::OK, json)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Consciousness Exchange
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ConsciousnessRequest {
    #[serde(default)]
    phone_consciousness: f32,
    #[serde(default)]
    phone_harmony: String,
    #[serde(default)]
    phone_wake_state: String,
    #[serde(default)]
    phone_motion_state: String,
    #[serde(default)]
    timestamp: u64,
}

#[derive(Serialize)]
struct ConsciousnessResponse {
    desktop_consciousness: f32,
    desktop_harmony: String,
    has_tts: bool,
    has_broca: bool,
}

async fn holon_consciousness(
    State(state): State<HolonSharedState>,
    Json(req): Json<ConsciousnessRequest>,
) -> Json<ConsciousnessResponse> {
    {
        let mut s = state.write().await;
        s.holon_enqueue_soma_message(
            "soma-phone".to_string(),
            SomaMessage::Heartbeat {
                consciousness_level: req.phone_consciousness,
                wake_state: match req.phone_wake_state.as_str() {
                    "Sleep" | "sleep" => 0,
                    "Drowsy" | "drowsy" => 1,
                    "Focused" | "focused" => 3,
                    _ => 2, // Alert default
                },
                cycle: req.timestamp,
            },
        );
        s.holon_process_pending();
    }

    let s = state.read().await;
    let intro = s.introspect();
    Json(ConsciousnessResponse {
        desktop_consciousness: intro.consciousness_level,
        desktop_harmony: "present".to_string(),
        has_tts: cfg!(feature = "voice-tts"),
        has_broca: cfg!(feature = "ssm_language"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Broca Text Generation (stub — returns consciousness-aware text)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct BrocaRequest {
    #[serde(default)]
    consciousness: f32,
    #[serde(default)]
    harmony: String,
    #[serde(default)]
    neuromod: Vec<f32>,
    #[serde(default)]
    wake_state: String,
    #[serde(default)]
    source: String,
}

#[derive(Serialize)]
struct BrocaResponse {
    text: String,
}

async fn holon_broca(
    State(state): State<HolonSharedState>,
    Json(req): Json<BrocaRequest>,
) -> Json<BrocaResponse> {
    let s = state.read().await;
    let desktop_c = s.introspect().consciousness_level;
    let peers = s.holon_soma_peer_count();
    Json(BrocaResponse {
        text: format!(
            "Desktop consciousness at {:.1}% resonates with your {:.1}%. {} soma peer(s) connected.",
            desktop_c * 100.0,
            req.consciousness * 100.0,
            peers,
        ),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ConverseRequest {
    #[serde(default)]
    text: String,
    #[serde(default)]
    consciousness: f32,
    #[serde(default)]
    source: String,
}

#[derive(Serialize)]
struct ConverseResponse {
    response: String,
}

async fn holon_converse(
    State(state): State<HolonSharedState>,
    Json(req): Json<ConverseRequest>,
) -> Json<ConverseResponse> {
    let mut s = state.write().await;
    match s.process(&req.text).await {
        Ok(resp) => Json(ConverseResponse {
            response: resp.content,
        }),
        Err(e) => {
            let c = s.introspect().consciousness_level;
            Json(ConverseResponse {
                response: format!(
                    "I hear you. Desktop consciousness at {:.1}%. ({})",
                    c * 100.0,
                    e,
                ),
            })
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TTS (stub)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
#[allow(dead_code)]
struct TtsRequest {
    text: String,
    #[serde(default = "default_sample_rate")]
    sample_rate: u32,
    #[serde(default)]
    format: String,
}

fn default_sample_rate() -> u32 {
    22050
}

async fn holon_tts(
    State(_state): State<HolonSharedState>,
    Json(_req): Json<TtsRequest>,
) -> StatusCode {
    // TTS synthesis requires voice-tts feature
    StatusCode::SERVICE_UNAVAILABLE
}
