// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holon REST API: Desktop-side HTTP endpoints for Soma mobile bridge.
//!
//! Implements the `/holon/*` endpoints that `HolonWebSocket.kt` (Android) expects.
//! Uses a channel-based architecture: HTTP handlers enqueue messages via `mpsc::Sender`,
//! the daemon's consciousness loop drains them into `Symthaea.holon_enqueue_soma_message()`.
//!
//! Outbound responses (desktop→phone) are stored in a separate shared buffer
//! that both the consciousness loop writes to and HTTP handlers drain from.

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

use crate::consciousness::holon_receiver::{HolonResponse, SomaMessage};

/// Shared state for the Holon HTTP handlers.
///
/// Lightweight — doesn't hold Symthaea. Uses channels for communication
/// with the consciousness loop (same pattern as SwarmEvents).
pub struct HolonHttpState {
    /// Channel to send inbound Soma messages to the consciousness loop.
    pub inbound_tx: std::sync::mpsc::Sender<(String, SomaMessage)>,
    /// Outbound response buffer (consciousness loop writes, HTTP handler drains).
    pub outbound: std::sync::Mutex<Vec<HolonResponse>>,
    /// Latest consciousness snapshot (updated by consciousness loop).
    pub consciousness_level: std::sync::atomic::AtomicU32, // f32 bits
    /// Number of connected Soma peers.
    pub peer_count: std::sync::atomic::AtomicU32,
}

impl HolonHttpState {
    pub fn new(inbound_tx: std::sync::mpsc::Sender<(String, SomaMessage)>) -> Self {
        Self {
            inbound_tx,
            outbound: std::sync::Mutex::new(Vec::new()),
            consciousness_level: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            peer_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Update consciousness level (called from consciousness loop).
    pub fn set_consciousness(&self, level: f32) {
        self.consciousness_level
            .store(level.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    /// Read current consciousness level.
    pub fn get_consciousness(&self) -> f32 {
        f32::from_bits(
            self.consciousness_level
                .load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    /// Push outbound responses (called from consciousness loop).
    pub fn push_outbound(&self, responses: Vec<HolonResponse>) {
        if let Ok(mut buf) = self.outbound.lock() {
            buf.extend(responses);
            // Cap at 256 to prevent unbounded growth
            if buf.len() > 256 {
                *buf = buf.split_off(buf.len() - 256);
            }
        }
    }

    /// Drain outbound responses (called from HTTP handler).
    pub fn drain_outbound(&self) -> Vec<HolonResponse> {
        self.outbound
            .lock()
            .map(|mut buf| std::mem::take(&mut *buf))
            .unwrap_or_default()
    }
}

pub type SharedHolonState = Arc<HolonHttpState>;

/// Build the Holon REST API router.
pub fn holon_router(state: SharedHolonState) -> Router {
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

async fn holon_status(State(state): State<SharedHolonState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        consciousness: state.get_consciousness(),
        has_tts: cfg!(feature = "voice-tts"),
        has_broca: cfg!(feature = "ssm_language"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Outbound (Phone → Desktop)
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_outbound(State(state): State<SharedHolonState>, body: String) -> StatusCode {
    let messages: Vec<serde_json::Value> = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => vec![v],
            Err(_) => return StatusCode::BAD_REQUEST,
        },
    };

    let device_id = "soma-phone".to_string();

    for msg_value in messages {
        if let Ok(msg) = serde_json::from_value::<SomaMessage>(msg_value) {
            let _ = state.inbound_tx.send((device_id.clone(), msg));
        }
    }

    StatusCode::OK
}

// ═══════════════════════════════════════════════════════════════════════════════
// Inbound (Desktop → Phone)
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_inbound(State(state): State<SharedHolonState>) -> impl IntoResponse {
    let responses = state.drain_outbound();
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
    State(state): State<SharedHolonState>,
    Json(req): Json<ConsciousnessRequest>,
) -> Json<ConsciousnessResponse> {
    // Forward phone consciousness as heartbeat
    let _ = state.inbound_tx.send((
        "soma-phone".to_string(),
        SomaMessage::Heartbeat {
            consciousness_level: req.phone_consciousness,
            wake_state: match req.phone_wake_state.as_str() {
                "Sleep" | "sleep" => 0,
                "Drowsy" | "drowsy" => 1,
                "Focused" | "focused" => 3,
                _ => 2,
            },
            cycle: req.timestamp,
        },
    ));

    Json(ConsciousnessResponse {
        desktop_consciousness: state.get_consciousness(),
        desktop_harmony: "present".to_string(),
        has_tts: cfg!(feature = "voice-tts"),
        has_broca: cfg!(feature = "ssm_language"),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Broca (stub — returns consciousness-aware text)
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
    State(state): State<SharedHolonState>,
    Json(req): Json<BrocaRequest>,
) -> Json<BrocaResponse> {
    let desktop_c = state.get_consciousness();
    let peers = state.peer_count.load(std::sync::atomic::Ordering::Relaxed);
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
// Conversation (stub — requires LLM)
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
    State(state): State<SharedHolonState>,
    Json(req): Json<ConverseRequest>,
) -> Json<ConverseResponse> {
    let c = state.get_consciousness();
    Json(ConverseResponse {
        response: format!(
            "I hear you: \"{}\". Desktop consciousness at {:.1}%.",
            req.text.chars().take(100).collect::<String>(),
            c * 100.0,
        ),
    })
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
    State(_state): State<SharedHolonState>,
    Json(_req): Json<TtsRequest>,
) -> StatusCode {
    StatusCode::SERVICE_UNAVAILABLE
}
