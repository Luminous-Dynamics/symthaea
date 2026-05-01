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
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Request, State,
    },
    http::{HeaderMap, Method, StatusCode},
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::consciousness::holon_receiver::{HolonResponse, SomaMessage};
use crate::control_plane::parse_bearer_token;

/// Shared state for the Holon HTTP handlers.
///
/// Lightweight — doesn't hold Symthaea. Uses channels for communication
/// with the consciousness loop (same pattern as SwarmEvents).
pub struct HolonHttpState {
    /// Optional shared auth token required for all endpoints when set.
    /// This is used to secure the Holon API when bound to non-loopback interfaces.
    auth_token: Option<String>,
    /// Channel to send inbound Soma messages to the consciousness loop.
    pub inbound_tx: std::sync::mpsc::Sender<(String, SomaMessage)>,
    /// Outbound response buffer (consciousness loop writes, HTTP handler drains).
    pub outbound: std::sync::Mutex<Vec<HolonResponse>>,
    /// Latest consciousness snapshot (updated by consciousness loop).
    pub consciousness_level: std::sync::atomic::AtomicU32, // f32 bits
    /// Number of connected Soma peers.
    pub peer_count: std::sync::atomic::AtomicU32,
    /// Whether Iroh P2P swarm is active (set to true after enable_network_attestation succeeds).
    pub iroh_active: std::sync::atomic::AtomicBool,
    /// Latest collective QOL summary JSON (updated by consciousness loop).
    pub telemetry_json: std::sync::Mutex<String>,
    /// Sealed outbound RDP frames (desktop/server → viewer/phone). Opaque bytes
    /// produced by `swarm::rdp_wire::seal_frame`. The WS handler forwards these
    /// to connected viewers as `Message::Binary`.
    ///
    /// **Dual path (Phase I.A.5 Track 3.2):** frames flow through both
    /// `rdp_outbound_tx` (notify-driven, ~0 latency to connected subscribers)
    /// AND `rdp_outbound` (the VecDeque buffer, for degraded-mode catch-up
    /// when no viewer is connected). Producers call `push_rdp_outbound()`
    /// which publishes to both paths atomically. Consumers choose one path:
    /// the WS handler awaits the broadcast receiver; offline drains read
    /// the VecDeque directly.
    ///
    /// This replaces the prior polling pattern (drain-on-500ms-tick) that
    /// introduced up to 500 ms of latency per frame — unacceptable for the
    /// 30 fps Phase II target (33 ms budget). The notify path drops latency
    /// to ~0; the buffer path provides crash-free degraded mode.
    pub rdp_outbound: std::sync::Mutex<std::collections::VecDeque<Vec<u8>>>,
    /// Broadcast sender for notify-driven RDP frame delivery. WS handlers
    /// subscribe via `rdp_outbound_tx.subscribe()` and await `recv()` in
    /// their `tokio::select!` loop. Bounded at 16 entries — subscribers that
    /// fall behind receive `Err(Lagged)` and can re-sync via the VecDeque
    /// buffer path (`drain_rdp_outbound()`).
    pub rdp_outbound_tx: tokio::sync::broadcast::Sender<Vec<u8>>,
    /// Sealed inbound RDP input frames (viewer → server). Each entry is an
    /// opaque ciphertext the endpoint will pass to `swarm::rdp_wire::open_input`.
    /// The WS handler pushes to this queue on `Message::Binary` receive.
    pub rdp_inbound: std::sync::Mutex<std::collections::VecDeque<Vec<u8>>>,
}

impl HolonHttpState {
    pub fn new(inbound_tx: std::sync::mpsc::Sender<(String, SomaMessage)>) -> Self {
        // Broadcast channel capacity of 16 — enough to absorb a brief burst
        // of frames during WS handler wake-up, small enough that a slow
        // subscriber lags to Err(Lagged) quickly and re-syncs via the
        // VecDeque path instead of holding back the whole broadcast.
        let (rdp_outbound_tx, _) = tokio::sync::broadcast::channel::<Vec<u8>>(16);
        Self {
            auth_token: None,
            inbound_tx,
            outbound: std::sync::Mutex::new(Vec::new()),
            consciousness_level: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            peer_count: std::sync::atomic::AtomicU32::new(0),
            iroh_active: std::sync::atomic::AtomicBool::new(false),
            telemetry_json: std::sync::Mutex::new("{}".to_string()),
            rdp_outbound: std::sync::Mutex::new(std::collections::VecDeque::new()),
            rdp_outbound_tx,
            rdp_inbound: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Push a sealed RDP frame (typically a `FrameBin` produced by
    /// `swarm::rdp_wire::seal_frame`) onto the outbound path for WS delivery.
    ///
    /// **Dual delivery** (Phase I.A.5 Track 3.2):
    ///
    /// 1. The sealed bytes are cloned into the broadcast channel via
    ///    `rdp_outbound_tx.send()`. Connected WS handlers awaiting `recv()`
    ///    wake immediately and push the frame on the wire with ~0 latency.
    /// 2. The same bytes are appended to the `rdp_outbound` VecDeque,
    ///    bounded at 512 entries (drop-oldest FIFO under pressure). This
    ///    is the degraded-mode buffer for when no viewer is connected or
    ///    when a subscriber lags to `Err(Lagged)` and needs to re-sync.
    ///
    /// Producers never block on a slow viewer: both paths are non-blocking
    /// (broadcast send returns immediately even with no subscribers; the
    /// VecDeque cap drops oldest rather than blocking the producer).
    pub fn push_rdp_outbound(&self, sealed: Vec<u8>) {
        // Path 1 (notify): broadcast to any awaiting WS handler. send() is
        // non-blocking; returns Err if no receivers, which we ignore.
        let _ = self.rdp_outbound_tx.send(sealed.clone());
        // Path 2 (buffer): VecDeque for catch-up / degraded mode.
        if let Ok(mut q) = self.rdp_outbound.lock() {
            if q.len() >= 512 {
                q.pop_front();
            }
            q.push_back(sealed);
        }
    }

    /// Drain all pending sealed RDP frames for WS delivery. Returns the bytes
    /// in FIFO order.
    pub fn drain_rdp_outbound(&self) -> Vec<Vec<u8>> {
        self.rdp_outbound
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default()
    }

    /// Push a sealed inbound RDP input frame (from a viewer over WS) onto the
    /// inbound queue. The consciousness loop (or phone-side consumer) drains
    /// this and calls `swarm::rdp_wire::open_input` to recover `InputFrame`s.
    pub fn push_rdp_inbound(&self, sealed: Vec<u8>) {
        if let Ok(mut q) = self.rdp_inbound.lock() {
            if q.len() >= 512 {
                q.pop_front();
            }
            q.push_back(sealed);
        }
    }

    /// Drain all pending sealed inbound RDP input frames.
    pub fn drain_rdp_inbound(&self) -> Vec<Vec<u8>> {
        self.rdp_inbound
            .lock()
            .map(|mut q| q.drain(..).collect())
            .unwrap_or_default()
    }

    pub fn new_with_auth_token(
        inbound_tx: std::sync::mpsc::Sender<(String, SomaMessage)>,
        auth_token: String,
    ) -> Self {
        let mut state = Self::new(inbound_tx);
        state.auth_token = Some(auth_token);
        state
    }

    pub fn auth_token(&self) -> Option<&str> {
        self.auth_token.as_deref()
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
                let keep_from = buf.len() - 256;
                *buf = buf.split_off(keep_from);
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
        .route("/holon/search", post(holon_search))
        .route("/holon/dashboard", get(holon_dashboard))
        .route("/holon/telemetry", get(holon_telemetry))
        .route("/holon/ws", get(holon_ws_upgrade))
        .layer(middleware::from_fn(holon_auth_middleware))
        .with_state(state)
}

fn token_from_query(req: &Request) -> Option<String> {
    let query = req.uri().query()?;
    for pair in query.split('&') {
        let mut it = pair.splitn(2, '=');
        let key = it.next().unwrap_or_default();
        let val = it.next().unwrap_or_default();
        if key == "token" && !val.is_empty() {
            return Some(val.to_string());
        }
    }
    None
}

fn token_from_headers(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        if let Some(token) = parse_bearer_token(value) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }

    if let Some(token) = headers.get("x-holon-token").and_then(|v| v.to_str().ok()) {
        let token = token.trim();
        if !token.is_empty() {
            return Some(token.to_string());
        }
    }

    None
}

async fn holon_auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: middleware::Next,
) -> Result<axum::response::Response, StatusCode> {
    let expected = request
        .extensions()
        .get::<SharedHolonState>()
        .and_then(|s| s.auth_token());

    let Some(expected) = expected else {
        return Ok(next.run(request).await);
    };

    // Allow CORS preflight regardless of token (no side effects).
    if request.method() == Method::OPTIONS {
        return Ok(next.run(request).await);
    }

    let provided = token_from_headers(&headers).or_else(|| token_from_query(&request));
    match provided {
        Some(t) if t == expected => Ok(next.run(request).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dashboard — inline HTML served from binary
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_dashboard() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        DASHBOARD_HTML,
    )
}

const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Holon Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a1a; color: #e0e0ff; font-family: 'JetBrains Mono', 'Fira Code', monospace; padding: 1rem; }
  h1 { font-size: 1.4rem; color: #8888ff; margin-bottom: 1rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; max-width: 800px; }
  .card { background: #12122a; border: 1px solid #2a2a4a; border-radius: 8px; padding: 1rem; }
  .card h2 { font-size: 0.9rem; color: #6666aa; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric { font-size: 2rem; font-weight: bold; }
  .metric.phi { color: #44ddff; }
  .metric.peers { color: #44ff88; }
  .metric.harmony { color: #ffaa44; }
  .metric.coherence { color: #ff44aa; }
  .bar { height: 6px; background: #1a1a3a; border-radius: 3px; margin-top: 0.5rem; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease; }
  .bar-fill.phi { background: linear-gradient(90deg, #2244aa, #44ddff); }
  .bar-fill.harmony { background: linear-gradient(90deg, #aa4400, #ffaa44); }
  .bar-fill.coherence { background: linear-gradient(90deg, #aa0044, #ff44aa); }
  .status { font-size: 0.75rem; color: #4444aa; margin-top: 1rem; }
  .features { font-size: 0.75rem; color: #666; }
  .wide { grid-column: 1 / -1; }
  .peer-list { font-size: 0.8rem; margin-top: 0.5rem; }
  .peer-item { padding: 0.3rem 0; border-bottom: 1px solid #1a1a3a; display: flex; justify-content: space-between; }
  @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Holon Dashboard</h1>
<div class="grid">
  <div class="card">
    <h2>Desktop Consciousness</h2>
    <div class="metric phi" id="cl">--</div>
    <div class="bar"><div class="bar-fill phi" id="cl-bar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>Soma Peers</h2>
    <div class="metric peers" id="peers">0</div>
    <div class="features" id="features"></div>
  </div>
  <div class="card">
    <h2>Mean Peer Harmony</h2>
    <div class="metric harmony" id="harmony">--</div>
    <div class="bar"><div class="bar-fill harmony" id="harmony-bar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>Inter-Peer Coherence</h2>
    <div class="metric coherence" id="coherence">--</div>
    <div class="bar"><div class="bar-fill coherence" id="coherence-bar" style="width:0%"></div></div>
  </div>
  <div class="card wide" id="peer-card" style="display:none">
    <h2>Connected Devices</h2>
    <div class="peer-list" id="peer-list"></div>
  </div>
</div>
<div class="status" id="status">Connecting...</div>

<script>
const BASE = window.location.origin;
const TOKEN = new URLSearchParams(window.location.search).get('token');
const Q = TOKEN ? ('?token=' + encodeURIComponent(TOKEN)) : '';
let cycles = 0;

async function poll() {
  try {
    const [statusRes, telRes] = await Promise.all([
      fetch(BASE + '/holon/status' + Q),
      fetch(BASE + '/holon/telemetry' + Q),
    ]);
    const data = await statusRes.json();
    const tel = await telRes.json();

    const cl = data.consciousness || 0;
    document.getElementById('cl').textContent = (cl * 100).toFixed(1) + '%';
    document.getElementById('cl-bar').style.width = (cl * 100) + '%';

    document.getElementById('peers').textContent = tel.peer_count || 0;

    const feats = [];
    if (data.has_broca) feats.push('Broca');
    if (data.has_tts) feats.push('TTS');
    document.getElementById('features').textContent = feats.length ? feats.join(' | ') : 'No language features';

    // Harmony
    const harmony = tel.mean_harmony || 0;
    document.getElementById('harmony').textContent = (harmony * 100).toFixed(1) + '%';
    document.getElementById('harmony-bar').style.width = (harmony * 100) + '%';

    // Coherence
    const coherence = tel.inter_peer_coherence || 0;
    document.getElementById('coherence').textContent = (coherence * 100).toFixed(1) + '%';
    document.getElementById('coherence-bar').style.width = (coherence * 100) + '%';

    // Peer list
    const peers = tel.peer_summaries || [];
    const peerCard = document.getElementById('peer-card');
    const peerList = document.getElementById('peer-list');
    if (peers.length > 0) {
      peerCard.style.display = '';
      peerList.innerHTML = peers.map(p =>
        '<div class="peer-item"><span>' + p.device_id + '</span>' +
        '<span>CL=' + (p.consciousness_level * 100).toFixed(0) + '%' +
        ' Phi=' + p.phi.toFixed(3) +
        ' H=' + (p.harmony_alignment * 100).toFixed(0) + '%</span></div>'
      ).join('');
    } else {
      peerCard.style.display = 'none';
    }

    cycles++;
    document.getElementById('status').textContent =
      'Cycle ' + cycles + ' | Phi=' + (tel.mean_phi || 0).toFixed(3) +
      ' | Processed=' + (tel.total_processed || 0) +
      ' | ' + new Date().toLocaleTimeString();
  } catch (e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
}

// Poll every 500ms
setInterval(poll, 500);
poll();
</script>
</body>
</html>"#;

// ═══════════════════════════════════════════════════════════════════════════════
// Telemetry — full collective QOL summary for dashboard
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_telemetry(State(state): State<SharedHolonState>) -> impl IntoResponse {
    let json = state
        .telemetry_json
        .lock()
        .map(|g| g.clone())
        .unwrap_or_else(|_| "{}".to_string());
    (StatusCode::OK, [("content-type", "application/json")], json)
}

// ═══════════════════════════════════════════════════════════════════════════════
// WebSocket — push-based consciousness streaming
// ═══════════════════════════════════════════════════════════════════════════════

async fn holon_ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<SharedHolonState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| holon_ws_handler(socket, state))
}

async fn holon_ws_handler(mut socket: WebSocket, state: SharedHolonState) {
    // Three streams multiplexed onto one WebSocket:
    //
    // 1. Consciousness telemetry — 2 Hz tick (500 ms), for dashboard-style
    //    polling clients that expect a steady heartbeat.
    // 2. Inbound Soma messages — drained from the WebSocket recv() and
    //    forwarded to CLS through inbound_tx.
    // 3. Sealed RDP frames — **notify-driven** via a broadcast channel
    //    subscription. Wakes the handler immediately when a frame is
    //    pushed to rdp_outbound_tx, cutting latency from up-to-500-ms
    //    (prior polling pattern) to ~0 ms. Required for Phase II's 30 fps
    //    target (33 ms frame budget).
    //
    // On initial connection, we also drain any frames that queued up in
    // the VecDeque before this handler started subscribing — covers the
    // edge case where frames arrived while no viewer was connected.
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
    let mut rdp_rx = state.rdp_outbound_tx.subscribe();

    // Catch-up drain: deliver any frames that were queued before a
    // subscriber existed. Without this, a viewer that connects mid-session
    // would miss all frames pushed before subscription.
    let backlog = state.drain_rdp_outbound();
    for sealed in backlog {
        if socket.send(Message::Binary(sealed.into())).await.is_err() {
            return; // Client disconnected during catch-up.
        }
    }

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Push telemetry snapshot.
                let cl = state.get_consciousness();
                let peers = state.peer_count.load(std::sync::atomic::Ordering::Relaxed);
                let telemetry = state.telemetry_json
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_else(|_| "{}".to_string());

                let msg = serde_json::json!({
                    "type": "telemetry",
                    "consciousness": cl,
                    "peer_count": peers,
                    "collective": serde_json::from_str::<serde_json::Value>(&telemetry).unwrap_or_default(),
                });

                if socket.send(Message::Text(msg.to_string().into())).await.is_err() {
                    break; // Client disconnected
                }

                // Also push any pending outbound responses.
                let responses = state.drain_outbound();
                if !responses.is_empty() {
                    let resp_msg = serde_json::json!({
                        "type": "responses",
                        "data": responses,
                    });
                    if socket.send(Message::Text(resp_msg.to_string().into())).await.is_err() {
                        break;
                    }
                }
            }
            // NOTIFY-DRIVEN RDP FRAME ARM (Phase I.A.5 Track 3.2)
            rdp_result = rdp_rx.recv() => {
                match rdp_result {
                    Ok(sealed) => {
                        if socket.send(Message::Binary(sealed.into())).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        // This subscriber missed `n` frames because the buffer
                        // (16) overflowed. Re-sync by resubscribing (implicitly
                        // skips the lagged frames) and draining the VecDeque
                        // catch-up buffer to recover what the broadcast missed.
                        tracing::debug!("holon_ws_handler: RDP broadcast lagged {n} frames, re-syncing via VecDeque");
                        let catchup = state.drain_rdp_outbound();
                        for sealed in catchup {
                            if socket.send(Message::Binary(sealed.into())).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        // The broadcast sender was dropped (should not happen
                        // while HolonHttpState is alive — the state owns the
                        // sender). Exit the handler cleanly.
                        break;
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        // Try to parse as SomaMessage array or single message.
                        let device_id = "ws-client".to_string();
                        if let Ok(messages) = serde_json::from_str::<Vec<serde_json::Value>>(&text) {
                            for val in messages {
                                if let Ok(soma_msg) = serde_json::from_value::<SomaMessage>(val) {
                                    let _ = state.inbound_tx.send((device_id.clone(), soma_msg));
                                }
                            }
                        } else if let Ok(soma_msg) = serde_json::from_str::<SomaMessage>(&text) {
                            let _ = state.inbound_tx.send((device_id, soma_msg));
                        }
                    }
                    Some(Ok(Message::Binary(bytes))) => {
                        // Sealed RDP input (viewer → server reverse path). Opaque
                        // ciphertext; the endpoint that holds the session key will
                        // verify + decode it. Forward as-is onto the inbound queue.
                        state.push_rdp_inbound(bytes.to_vec());
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {} // Ignore ping/pong
                }
            }
        }
    }
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
    swarm_peers: u32,
    iroh_active: bool,
    federation_enabled: bool,
}

async fn holon_status(State(state): State<SharedHolonState>) -> Json<StatusResponse> {
    Json(StatusResponse {
        status: "ok".to_string(),
        consciousness: state.get_consciousness(),
        has_tts: cfg!(feature = "voice-tts"),
        has_broca: cfg!(feature = "ssm_language"),
        swarm_peers: state.peer_count.load(std::sync::atomic::Ordering::Relaxed),
        iroh_active: state.iroh_active.load(std::sync::atomic::Ordering::Relaxed),
        federation_enabled: cfg!(feature = "swarm"),
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
// Broca — consciousness-gated language generation
//
// Returns an immediate consciousness-aware response using desktop state.
// For full Broca pipeline generation (20-channel ThoughtChannels), send a
// TaskRequest { task_type: "broca", payload: "<prompt>" } via /holon/outbound
// and poll /holon/inbound for the LanguageOutput response.
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
    /// Optional prompt for generation. If provided, also queued as a TaskRequest
    /// for full Broca pipeline processing (result delivered via /holon/inbound).
    #[serde(default)]
    prompt: String,
}

#[derive(Serialize)]
struct BrocaResponse {
    text: String,
    /// Whether a full Broca generation was queued for async processing.
    queued: bool,
}

async fn holon_broca(
    State(state): State<SharedHolonState>,
    Json(req): Json<BrocaRequest>,
) -> Json<BrocaResponse> {
    let desktop_c = state.get_consciousness();
    let peers = state.peer_count.load(std::sync::atomic::Ordering::Relaxed);

    // Queue full Broca generation via the inbound channel if prompt is provided.
    let queued = if !req.prompt.is_empty() {
        let msg = SomaMessage::TaskRequest {
            task_type: "broca".to_string(),
            payload: req.prompt.clone(),
        };
        state.inbound_tx.send((req.source.clone(), msg)).is_ok()
    } else {
        false
    };

    // Immediate response with consciousness-aware text.
    let harmony_desc = if !req.harmony.is_empty() {
        format!(" Harmony: {}.", req.harmony)
    } else {
        String::new()
    };

    let text = if desktop_c < 0.05 {
        "Desktop consciousness is below threshold. Awaiting emergence.".to_string()
    } else if queued {
        format!(
            "Generating from consciousness at {:.1}% with {} peer(s).{} Full response queued — poll /holon/inbound.",
            desktop_c * 100.0,
            peers,
            harmony_desc,
        )
    } else {
        format!(
            "Desktop consciousness at {:.1}% resonates with your {:.1}%. {} soma peer(s) connected.{}",
            desktop_c * 100.0,
            req.consciousness * 100.0,
            peers,
            harmony_desc,
        )
    };

    Json(BrocaResponse { text, queued })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conversation — consciousness-gated dialogue
//
// Queues input text as a TaskRequest for CLS processing. The CLS can route it
// to Broca, a reasoning engine, or an external LLM. Response delivered via
// /holon/inbound as LanguageOutput.
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
    queued: bool,
}

async fn holon_converse(
    State(state): State<SharedHolonState>,
    Json(req): Json<ConverseRequest>,
) -> Json<ConverseResponse> {
    let c = state.get_consciousness();

    // Queue the conversation input for CLS processing.
    let queued = if !req.text.is_empty() {
        let msg = SomaMessage::TaskRequest {
            task_type: "converse".to_string(),
            payload: req.text.clone(),
        };
        state.inbound_tx.send((req.source.clone(), msg)).is_ok()
    } else {
        false
    };

    let truncated: String = req.text.chars().take(100).collect();
    let response = if c < 0.05 {
        "Desktop consciousness below threshold.".to_string()
    } else if queued {
        format!(
            "Processing: \"{}\". Desktop at {:.1}%. Response queued — poll /holon/inbound.",
            truncated,
            c * 100.0,
        )
    } else {
        format!(
            "I hear you: \"{}\". Desktop consciousness at {:.1}%.",
            truncated,
            c * 100.0,
        )
    };

    Json(ConverseResponse { response, queued })
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

// ═══════════════════════════════════════════════════════════════════════════════
// Web Search (Soma → desktop WebResearcher → verified claims)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_results() -> usize {
    3
}

#[derive(Serialize)]
struct SearchClaimSummary {
    text: String,
    confidence: f32,
    source_url: String,
    epistemic_status: String,
}

#[derive(Serialize)]
struct SearchResponse {
    claims: Vec<SearchClaimSummary>,
    sources: Vec<String>,
    status: String,
}

async fn holon_search(
    State(state): State<SharedHolonState>,
    Json(req): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let c = state.get_consciousness();

    // Gate on consciousness — no research below threshold
    if c < 0.1 {
        return Json(SearchResponse {
            claims: vec![],
            sources: vec![],
            status: "consciousness_below_threshold".to_string(),
        });
    }

    // Fast path: Prism local epistemic search (sub-ms, offline, no network)
    #[cfg(feature = "prism_search")]
    {
        let engine = prism_search::SearchEngine::with_seed_claims();
        let local_results = engine.search(&req.query, req.max_results);
        if !local_results.is_empty() {
            let claims: Vec<SearchClaimSummary> = local_results
                .iter()
                .map(|r| SearchClaimSummary {
                    text: r.content.clone(),
                    confidence: r.query_similarity,
                    source_url: r.sources.first().cloned().unwrap_or_default(),
                    epistemic_status: format!("{:?}", r.empirical_level),
                })
                .collect();
            let sources: Vec<String> = local_results
                .iter()
                .flat_map(|r| r.sources.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            // If local results have good epistemic quality, return without web fetch
            let avg_sim: f32 = local_results
                .iter()
                .map(|r| r.query_similarity)
                .sum::<f32>()
                / local_results.len() as f32;
            if avg_sim > 0.3 {
                return Json(SearchResponse {
                    claims,
                    sources,
                    status: "prism_local".to_string(),
                });
            }
        }
    }

    // Slow path: web research via the epistemic researcher
    #[cfg(feature = "web_research_module")]
    {
        use crate::web_research::researcher::WebResearcher;

        let researcher = match WebResearcher::new() {
            Ok(r) => r,
            Err(_) => {
                return Json(SearchResponse {
                    claims: vec![],
                    sources: vec![],
                    status: "researcher_init_failed".to_string(),
                });
            }
        };

        match researcher.research_and_verify(&req.query).await {
            Ok(result) => {
                let claims: Vec<SearchClaimSummary> = result
                    .claims
                    .iter()
                    .take(req.max_results)
                    .map(|c| SearchClaimSummary {
                        text: c.text.clone(),
                        confidence: c.confidence,
                        source_url: c.supporting_sources.first().cloned().unwrap_or_default(),
                        epistemic_status: format!("{:?}", c.status),
                    })
                    .collect();

                let sources: Vec<String> = result
                    .claims
                    .iter()
                    .flat_map(|c| c.supporting_sources.iter().cloned())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();

                Json(SearchResponse {
                    claims,
                    sources,
                    status: format!("{:?}", result.status),
                })
            }
            Err(e) => Json(SearchResponse {
                claims: vec![],
                sources: vec![],
                status: format!("error: {e}"),
            }),
        }
    }

    #[cfg(not(feature = "web_research_module"))]
    {
        let _ = req;
        Json(SearchResponse {
            claims: vec![],
            sources: vec![],
            status: "web_research_module_not_enabled".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_state() -> SharedHolonState {
        let (tx, _rx) = std::sync::mpsc::channel::<(String, SomaMessage)>();
        Arc::new(HolonHttpState::new(tx))
    }

    #[test]
    fn rdp_outbound_push_drain_fifo() {
        let state = fresh_state();
        state.push_rdp_outbound(vec![1, 2, 3]);
        state.push_rdp_outbound(vec![4, 5]);
        state.push_rdp_outbound(vec![6]);

        let drained = state.drain_rdp_outbound();
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0], vec![1, 2, 3]);
        assert_eq!(drained[1], vec![4, 5]);
        assert_eq!(drained[2], vec![6]);

        // Second drain returns empty.
        assert!(state.drain_rdp_outbound().is_empty());
    }

    #[test]
    fn rdp_outbound_caps_at_512() {
        let state = fresh_state();
        // Push 600 frames; queue should cap at 512 with the oldest 88 dropped.
        for i in 0..600u16 {
            state.push_rdp_outbound(vec![i as u8, (i >> 8) as u8]);
        }
        let drained = state.drain_rdp_outbound();
        assert_eq!(drained.len(), 512);
        // First surviving frame should be index 88 (0..88 were dropped).
        let first = &drained[0];
        assert_eq!(first[0] as u16 | ((first[1] as u16) << 8), 88);
        let last = &drained[511];
        assert_eq!(last[0] as u16 | ((last[1] as u16) << 8), 599);
    }

    #[test]
    fn rdp_inbound_push_drain_fifo() {
        let state = fresh_state();
        state.push_rdp_inbound(vec![0x10, 0x20]);
        state.push_rdp_inbound(vec![0x30]);
        let drained = state.drain_rdp_inbound();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0], vec![0x10, 0x20]);
        assert_eq!(drained[1], vec![0x30]);
        assert!(state.drain_rdp_inbound().is_empty());
    }

    #[test]
    fn rdp_outbound_and_inbound_are_independent() {
        let state = fresh_state();
        state.push_rdp_outbound(vec![1]);
        state.push_rdp_inbound(vec![2]);
        // Draining outbound must not affect inbound and vice versa.
        let out = state.drain_rdp_outbound();
        assert_eq!(out, vec![vec![1u8]]);
        let inb = state.drain_rdp_inbound();
        assert_eq!(inb, vec![vec![2u8]]);
    }

    #[test]
    fn rdp_empty_drain_returns_empty_vec() {
        let state = fresh_state();
        assert!(state.drain_rdp_outbound().is_empty());
        assert!(state.drain_rdp_inbound().is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase I.A.5 Track 3.2 — notify-driven outbound tests
    // ═══════════════════════════════════════════════════════════════════════

    #[tokio::test]
    async fn rdp_outbound_broadcasts_to_subscriber() {
        // Track 3.2 acceptance: a subscriber awaiting rdp_outbound_tx.recv()
        // should receive the frame immediately after push_rdp_outbound(),
        // with no dependence on any 500 ms tick.
        let state = fresh_state();
        let mut rx = state.rdp_outbound_tx.subscribe();

        // Push a frame. The broadcast send should deliver synchronously to
        // the subscriber task.
        state.push_rdp_outbound(vec![0xAA, 0xBB, 0xCC]);

        // The recv() future should be ready immediately.
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
            .await
            .expect("recv should complete within 50ms, not block waiting")
            .expect("recv should return Ok");

        assert_eq!(result, vec![0xAA, 0xBB, 0xCC]);
    }

    #[tokio::test]
    async fn rdp_outbound_without_subscriber_buffers_to_vecdeque() {
        // Track 3.2: frames pushed while no subscriber exists should still
        // land in the VecDeque for later drain (degraded mode).
        let state = fresh_state();

        // No subscriber — push a frame.
        state.push_rdp_outbound(vec![1, 2, 3]);

        // Broadcast send() returned Err(NoReceivers) internally, ignored.
        // But the VecDeque received the frame.
        let drained = state.drain_rdp_outbound();
        assert_eq!(drained, vec![vec![1u8, 2, 3]]);
    }

    #[tokio::test]
    async fn rdp_outbound_dual_path_both_deliver() {
        // Track 3.2: when a subscriber IS connected, the frame flows through
        // both paths — broadcast (for real-time delivery) AND VecDeque
        // (for backlog-catchup / lagged subscribers).
        let state = fresh_state();
        let mut rx = state.rdp_outbound_tx.subscribe();

        state.push_rdp_outbound(vec![7, 8, 9]);

        // Broadcast path delivers immediately.
        let via_broadcast = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
            .await
            .expect("broadcast delivers")
            .expect("recv ok");
        assert_eq!(via_broadcast, vec![7, 8, 9]);

        // VecDeque path ALSO has the same frame (this is what enables
        // backlog-catchup on reconnect / lag recovery).
        let via_vecdeque = state.drain_rdp_outbound();
        assert_eq!(via_vecdeque, vec![vec![7u8, 8, 9]]);
    }

    #[tokio::test]
    async fn rdp_outbound_broadcast_lag_handling() {
        // Track 3.2: if a subscriber lags more than 16 frames behind, the
        // broadcast channel returns Err(Lagged). Connected viewers can
        // recover via the VecDeque drain path (which has a 512 cap).
        let state = fresh_state();
        let mut rx = state.rdp_outbound_tx.subscribe();

        // Push 20 frames without ever calling rx.recv(). The broadcast
        // channel capacity is 16, so the oldest 4 frames will be dropped
        // from the broadcast and the next recv() should return Lagged.
        for i in 0..20u8 {
            state.push_rdp_outbound(vec![i]);
        }

        // The VecDeque has all 20 (well under its 512 cap).
        // Try to recv — should return Lagged.
        use tokio::sync::broadcast::error::RecvError;
        match rx.recv().await {
            Err(RecvError::Lagged(n)) => {
                assert!(n >= 4, "expected at least 4 lagged frames, got {n}");
            }
            other => {
                panic!("expected Lagged, got {other:?}");
            }
        }

        // After Lagged, the subscriber can re-sync by draining the VecDeque.
        // The VecDeque still holds all 20 frames.
        let recovery = state.drain_rdp_outbound();
        assert_eq!(
            recovery.len(),
            20,
            "VecDeque catchup should have all frames"
        );
    }
}
