// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shell IPC Client - Connect TUI Shell to Symthaea Service
//!
//! Provides Unix socket IPC for the shell to communicate with the symthaea
//! cognitive service, receiving real-time consciousness metrics and submitting
//! commands for Phi-gated execution.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐     Unix Socket      ┌──────────────────────┐
//! │ symthaea-shell  │◄──────────────────►  │ symthaea-service     │
//! │ (TUI Client)    │   MessagePack IPC    │ (CognitiveLoopService)│
//! └─────────────────┘                      └──────────────────────┘
//! ```
//!
//! ## Protocol
//!
//! - Requests/responses encoded with MessagePack (rmp-serde)
//! - Bidirectional: service pushes metrics, shell sends commands
//! - Automatic reconnection on disconnect

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::UnixStream;
use tokio::sync::watch;

const IPC_PROTOCOL_VERSION: u32 = 1;
const IPC_MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;

// ═══════════════════════════════════════════════════════════════════════════════
// CONNECTION STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Connection state for visual indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConnectionState {
    /// Not connected to service
    #[default]
    Disconnected,
    /// Attempting to connect
    Connecting,
    /// Connected and receiving metrics
    Connected,
    /// Connected but service is degraded
    Degraded,
    /// Connection lost, attempting reconnect
    Reconnecting,
}

impl ConnectionState {
    /// Get visual indicator for this state
    pub fn indicator(&self) -> &'static str {
        match self {
            Self::Disconnected => "○",
            Self::Connecting => "◐",
            Self::Connected => "●",
            Self::Degraded => "◑",
            Self::Reconnecting => "◒",
        }
    }

    /// Get color hint for this state (RGB)
    pub fn color(&self) -> (u8, u8, u8) {
        match self {
            Self::Disconnected => (128, 128, 128),
            Self::Connecting => (255, 200, 0),
            Self::Connected => (0, 200, 0),
            Self::Degraded => (255, 165, 0),
            Self::Reconnecting => (255, 200, 0),
        }
    }

    /// Get human-readable status text
    pub fn status_text(&self) -> &'static str {
        match self {
            Self::Disconnected => "Disconnected",
            Self::Connecting => "Connecting...",
            Self::Connected => "Connected",
            Self::Degraded => "Degraded",
            Self::Reconnecting => "Reconnecting...",
        }
    }

    /// Get label for this state (alias for status_text)
    pub fn label(&self) -> &'static str {
        self.status_text()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS SNAPSHOT
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared mental movie type for IPC transport
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MentalMovie {
    pub frames: Vec<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub path_length: usize,
    pub semantic_coherence: f32,
}

/// Snapshot of consciousness metrics from the service
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricsSnapshot {
    /// Current Phi (consciousness level) from CognitiveLoopService
    pub phi: f64,
    /// Coherence measure from CfC dynamics
    pub coherence: f64,
    /// Whether system is above consciousness threshold
    pub is_conscious: bool,
    /// Current cognitive depth (Reflex/Cortical/DeepThought)
    pub cognitive_depth: String,
    /// Current response strategy
    pub strategy: String,
    /// Flow state active
    pub in_flow: bool,
    /// Prediction error (recent average)
    pub prediction_error: f32,
    /// Emotional valence (-1 to 1)
    pub emotional_valence: f32,
    /// Emotional arousal (0 to 1)
    pub emotional_arousal: f32,
    /// Timestamp of this snapshot
    pub timestamp_ms: u64,
    /// Service uptime in seconds
    pub uptime_secs: u64,
    /// Total cycles processed
    pub total_cycles: u64,
    /// Overall consciousness level (derived from phi + coherence)
    pub consciousness_level: f64,
    /// IPC latency in milliseconds
    pub latency_ms: u64,
    /// The latest mental simulation result from the cognitive loop.
    #[cfg(feature = "vision-manifold")]
    pub mental_movie: Option<MentalMovie>,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            phi: 0.5,
            coherence: 0.5,
            is_conscious: true,
            cognitive_depth: "Cortical".to_string(),
            strategy: "Supportive".to_string(),
            in_flow: false,
            prediction_error: 0.2,
            emotional_valence: 0.0,
            emotional_arousal: 0.5,
            timestamp_ms: 0,
            uptime_secs: 0,
            total_cycles: 0,
            consciousness_level: 0.5,
            latency_ms: 0,
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC MESSAGES
// ═══════════════════════════════════════════════════════════════════════════════

/// Request from shell to service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcRequest {
    /// Protocol version handshake
    Hello { version: u32 },
    /// Subscribe to metrics stream
    SubscribeMetrics,
    /// Unsubscribe from metrics
    UnsubscribeMetrics,
    /// Execute a command (with Phi gating)
    Execute {
        command: String,
        require_phi: Option<f64>,
    },
    /// Get completions for input
    GetCompletions { input: String, cursor_pos: usize },
    /// Validate a command (dry run)
    Validate { command: String },
    /// Get current metrics snapshot
    GetMetrics,
    /// Ping for keepalive
    Ping,
}

/// Response from service to shell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcResponse {
    /// Protocol version acknowledgment
    HelloAck { server_version: u32 },
    /// Metrics update (pushed periodically)
    Metrics(MetricsSnapshot),
    /// Completion suggestions
    Completions(Vec<CompletionItem>),
    /// Command execution result
    ExecutionResult {
        success: bool,
        output: String,
        phi_at_execution: f64,
        vetoed: bool,
        veto_reason: Option<String>,
    },
    /// Validation result
    ValidationResult {
        valid: bool,
        safety_level: String,
        preview: Option<String>,
        warnings: Vec<String>,
    },
    /// Subscription acknowledgment
    Subscribed,
    /// Pong response
    Pong,
    /// Error response
    Error(String),
}

/// Completion item from service
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionItem {
    /// Completion text
    pub text: String,
    /// Display label
    pub label: String,
    /// Kind of completion
    pub kind: String,
    /// HDC similarity score
    pub similarity: f32,
    /// Documentation/description
    pub docs: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIRE PROTOCOL - REQUEST/RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Request enum for the wire protocol
///
/// These are the primary request types used in the IPC wire protocol.
/// Serialized with MessagePack (rmp-serde) for efficient binary transport.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Request {
    /// Ping for keepalive / health check
    Ping,
    /// Get current status (phi, coherence, consciousness level)
    GetStatus,
    /// Execute a command with optional Phi gating
    Execute {
        command: String,
        phi_required: f64,
        dry_run: bool,
    },
    /// Get IntelliSense completions
    GetIntelliSense {
        input: String,
        cursor_pos: usize,
        context: ShellContextData,
    },
    /// Validate a command without executing
    Validate { command: String, dry_run: bool },
    /// Subscribe to metrics stream
    SubscribeMetrics,
    /// Unsubscribe from metrics stream
    UnsubscribeMetrics,
}

/// Response enum for the wire protocol
///
/// These are the primary response types used in the IPC wire protocol.
/// Serialized with MessagePack (rmp-serde) for efficient binary transport.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Response {
    /// Pong response to Ping request
    Pong { timestamp_ms: u64 },
    /// Status response with consciousness metrics
    Status {
        phi: f64,
        coherence: f64,
        consciousness_level: f64,
        is_conscious: bool,
        uptime_secs: u64,
    },
    /// IntelliSense result with completions
    IntelliSenseResult {
        completions: Vec<CompletionItem>,
        command_preview: Option<String>,
        phi: f64,
        confidence: f32,
    },
    /// Validation result
    ValidationResult {
        valid: bool,
        safety_level: SafetyLevelData,
        phi_required: f64,
        warnings: Vec<String>,
    },
    /// Execution result
    ExecutionResult {
        executed: bool,
        output: String,
        phi_at_execution: f64,
        gate_reason: Option<String>,
    },
    /// Metrics update (pushed periodically when subscribed)
    Metrics(MetricsSnapshot),
    /// Subscription acknowledgment
    Subscribed,
    /// Error response
    Error { code: i32, message: String },
}

/// Safety level data for command validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyLevelData {
    /// Safety level name (e.g., "GREEN", "YELLOW", "RED")
    pub level: String,
    /// Color hint for UI display
    pub color: String,
    /// Human-readable description
    pub description: String,
}

impl Default for SafetyLevelData {
    fn default() -> Self {
        Self {
            level: "GREEN".to_string(),
            color: "green".to_string(),
            description: "Safe operation".to_string(),
        }
    }
}

/// Shell context data for IntelliSense requests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ShellContextData {
    /// Current working directory
    pub cwd: String,
    /// Recent command history
    pub history: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIRE FRAMING - REQUEST/RESPONSE ENVELOPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Request envelope for wire framing
///
/// Wraps a Request with an optional correlation ID for request/response matching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestEnvelope {
    /// Optional correlation ID for request/response matching
    pub id: Option<String>,
    /// The actual request
    pub request: Request,
}

impl RequestEnvelope {
    /// Create a new request envelope without an ID
    pub fn new(request: Request) -> Self {
        Self { id: None, request }
    }

    /// Create a new request envelope with an ID
    pub fn with_id(id: String, request: Request) -> Self {
        Self {
            id: Some(id),
            request,
        }
    }
}

/// Response envelope for wire framing
///
/// Wraps a Response with the correlation ID from the request and latency info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseEnvelope {
    /// Correlation ID from the request (if provided)
    pub id: Option<String>,
    /// Latency in milliseconds (server-side processing time)
    pub latency_ms: Option<u64>,
    /// The actual response
    pub response: Response,
}

impl ResponseEnvelope {
    /// Create a new response envelope
    pub fn new(id: Option<String>, response: Response) -> Self {
        Self {
            id,
            latency_ms: None,
            response,
        }
    }

    /// Create a new response envelope with latency
    pub fn with_latency(id: Option<String>, latency_ms: u64, response: Response) -> Self {
        Self {
            id,
            latency_ms: Some(latency_ms),
            response,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIRE PROTOCOL TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Wire protocol for serializing/deserializing IPC messages
///
/// Supports both MessagePack (binary, efficient) and JSON (text, debugging).
pub struct WireProtocol;

impl WireProtocol {
    /// Encode a request envelope to MessagePack bytes
    pub fn encode_request(envelope: &RequestEnvelope) -> Result<Vec<u8>> {
        rmp_serde::to_vec(envelope).context("Failed to encode request to MessagePack")
    }

    /// Decode a request envelope from MessagePack bytes
    pub fn decode_request(bytes: &[u8]) -> Result<RequestEnvelope> {
        rmp_serde::from_slice(bytes).context("Failed to decode request from MessagePack")
    }

    /// Encode a response envelope to MessagePack bytes
    pub fn encode_response(envelope: &ResponseEnvelope) -> Result<Vec<u8>> {
        rmp_serde::to_vec(envelope).context("Failed to encode response to MessagePack")
    }

    /// Decode a response envelope from MessagePack bytes
    pub fn decode_response(bytes: &[u8]) -> Result<ResponseEnvelope> {
        rmp_serde::from_slice(bytes).context("Failed to decode response from MessagePack")
    }

    /// Encode a request envelope to JSON string (for debugging)
    pub fn encode_request_json(envelope: &RequestEnvelope) -> Result<String> {
        serde_json::to_string(envelope).context("Failed to encode request to JSON")
    }

    /// Decode a request envelope from JSON string
    pub fn decode_request_json(json: &str) -> Result<RequestEnvelope> {
        serde_json::from_str(json).context("Failed to decode request from JSON")
    }

    /// Encode a response envelope to JSON string (for debugging)
    pub fn encode_response_json(envelope: &ResponseEnvelope) -> Result<String> {
        serde_json::to_string(envelope).context("Failed to encode response to JSON")
    }

    /// Decode a response envelope from JSON string
    pub fn decode_response_json(json: &str) -> Result<ResponseEnvelope> {
        serde_json::from_str(json).context("Failed to decode response from JSON")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC CLIENT CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// IPC client configuration
#[derive(Debug, Clone)]
pub struct IpcClientConfig {
    /// Socket path (default: discover from XDG_RUNTIME_DIR)
    pub socket_path: Option<PathBuf>,
    /// Reconnect interval on disconnect
    pub reconnect_interval: Duration,
    /// Maximum reconnect attempts before giving up
    pub max_reconnect_attempts: u32,
    /// Metrics push interval (requested from service)
    pub metrics_interval: Duration,
    /// Request timeout
    pub request_timeout: Duration,
}

impl Default for IpcClientConfig {
    fn default() -> Self {
        Self {
            socket_path: None,
            reconnect_interval: Duration::from_secs(2),
            max_reconnect_attempts: 10,
            metrics_interval: Duration::from_millis(100),
            request_timeout: Duration::from_secs(5),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SOCKET DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════════

/// Discover the symthaea service socket
pub fn discover_socket() -> Option<PathBuf> {
    // Try XDG_RUNTIME_DIR first (preferred)
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let socket = PathBuf::from(runtime_dir).join("symthaea/symthaea.sock");
        if socket.exists() {
            return Some(socket);
        }
    }

    // Try /run/user/$UID
    if let Ok(uid) = std::env::var("UID") {
        let socket = PathBuf::from(format!("/run/user/{uid}/symthaea/symthaea.sock"));
        if socket.exists() {
            return Some(socket);
        }
    }

    // Try /tmp fallback
    let tmp_socket = PathBuf::from("/tmp/symthaea.sock");
    if tmp_socket.exists() {
        return Some(tmp_socket);
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHELL IPC CLIENT
// ═══════════════════════════════════════════════════════════════════════════════

/// IPC client for shell-to-service communication
pub struct ShellIpcClient {
    /// Configuration
    config: IpcClientConfig,
    /// Current connection state
    state: ConnectionState,
    /// Socket path being used
    socket_path: Option<PathBuf>,
    /// Metrics watch sender (for push updates)
    metrics_tx: watch::Sender<MetricsSnapshot>,
    /// Last metrics snapshot
    last_metrics: MetricsSnapshot,
    /// Reconnect attempt counter
    reconnect_attempts: u32,
    /// Last successful ping time
    last_ping: Option<Instant>,
    /// Buffered reader for the socket read half
    reader: Option<BufReader<OwnedReadHalf>>,
    /// Buffered writer for the socket write half
    writer: Option<BufWriter<OwnedWriteHalf>>,
    /// Batch of pending requests
    batch: Vec<Request>,
    /// Last heartbeat timestamp
    last_heartbeat: Option<Instant>,
    /// Health threshold (max time since last successful heartbeat)
    health_threshold: Duration,
}

impl ShellIpcClient {
    /// Create a new IPC client with default configuration
    pub fn new() -> Self {
        Self::with_config(IpcClientConfig::default())
    }

    /// Create a new IPC client with specific configuration
    pub fn with_config(config: IpcClientConfig) -> Self {
        let (metrics_tx, _) = watch::channel(MetricsSnapshot::default());

        Self {
            config,
            state: ConnectionState::Disconnected,
            socket_path: None,
            metrics_tx,
            last_metrics: MetricsSnapshot::default(),
            reconnect_attempts: 0,
            last_ping: None,
            reader: None,
            writer: None,
            batch: Vec::new(),
            last_heartbeat: None,
            health_threshold: Duration::from_secs(30),
        }
    }

    /// Create with default configuration (alias for new)
    pub fn with_defaults() -> Self {
        Self::new()
    }

    /// Create with a specific socket path
    pub fn with_socket_path(path: &std::path::Path) -> Self {
        let config = IpcClientConfig {
            socket_path: Some(path.to_path_buf()),
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Get current status (phi, coherence, is_conscious) from server
    pub async fn get_status(&mut self) -> Result<(f64, f64, bool)> {
        let response = self.send_request(Request::GetStatus).await?;
        match response {
            Response::Status {
                phi,
                coherence,
                is_conscious,
                ..
            } => Ok((phi, coherence, is_conscious)),
            Response::Error { code, message } => {
                anyhow::bail!("GetStatus failed: {message} (code {code})")
            }
            _ => {
                anyhow::bail!("Unexpected response to GetStatus request")
            }
        }
    }

    /// Get current status from cache (no network call) - convenience method
    pub fn get_status_cached(&self) -> (f64, f64, bool) {
        let m = &self.last_metrics;
        (m.phi, m.coherence, m.is_conscious)
    }

    /// Connect with automatic retry
    pub async fn connect_with_retry(&mut self, max_retries: u32) -> Result<()> {
        for attempt in 0..max_retries {
            match self.connect().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if attempt + 1 < max_retries {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        anyhow::bail!("Failed to connect after {max_retries} attempts")
    }

    /// Get current connection state
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Get a watch receiver for metrics updates
    pub fn metrics_receiver(&self) -> watch::Receiver<MetricsSnapshot> {
        self.metrics_tx.subscribe()
    }

    /// Subscribe to metrics updates with specified interval
    /// Returns a watch receiver that will receive updates at the given interval
    pub async fn subscribe_metrics_watch(
        &self,
        _interval_ms: u64,
    ) -> Result<watch::Receiver<MetricsSnapshot>> {
        // Return a subscription to the metrics channel
        // The interval is handled by the service-side push mechanism
        Ok(self.metrics_tx.subscribe())
    }

    /// Get the last metrics snapshot
    pub fn last_metrics(&self) -> &MetricsSnapshot {
        &self.last_metrics
    }

    /// Attempt to connect to the service
    pub async fn connect(&mut self) -> Result<()> {
        self.state = ConnectionState::Connecting;

        // Discover socket if not configured
        let socket_path = self
            .config
            .socket_path
            .clone()
            .or_else(discover_socket)
            .context("No symthaea socket found. Is the service running?")?;

        self.socket_path = Some(socket_path.clone());

        // Connect to socket
        let stream = UnixStream::connect(&socket_path)
            .await
            .with_context(|| format!("Failed to connect to {}", socket_path.display()))?;

        // Split the stream into read and write halves with buffering
        let (read_half, write_half) = stream.into_split();
        self.reader = Some(BufReader::new(read_half));
        self.writer = Some(BufWriter::new(write_half));

        self.handshake().await.context("IPC handshake failed")?;

        self.state = ConnectionState::Connected;
        self.reconnect_attempts = 0;
        self.last_ping = Some(Instant::now());
        self.last_heartbeat = Some(Instant::now());

        tracing::info!(socket = %socket_path.display(), "Connected to symthaea service");
        Ok(())
    }

    /// Disconnect from the service
    pub fn disconnect(&mut self) {
        self.state = ConnectionState::Disconnected;
        self.socket_path = None;
        self.reader = None;
        self.writer = None;
        self.batch.clear();
    }

    /// Update metrics from a snapshot (called when metrics received)
    pub fn update_metrics(&mut self, metrics: MetricsSnapshot) {
        self.last_metrics = metrics.clone();
        let _ = self.metrics_tx.send(metrics);
    }

    /// Simulate metrics for when service is not available
    pub fn simulate_metrics(&mut self, phi: f64, coherence: f64) {
        let snapshot = MetricsSnapshot {
            phi,
            coherence,
            is_conscious: phi > 0.3,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            ..Default::default()
        };
        self.update_metrics(snapshot);
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        matches!(self.state, ConnectionState::Connected)
    }

    /// Get socket path
    pub fn socket_path(&self) -> Option<&Path> {
        self.socket_path.as_deref()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // WIRE PROTOCOL METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Send a ping request and wait for pong response
    pub async fn ping(&mut self) -> Result<Response> {
        let response = self.send_request(Request::Ping).await?;
        self.last_ping = Some(Instant::now());
        Ok(response)
    }

    /// Send a heartbeat (ping) for keepalive
    pub async fn heartbeat(&mut self) -> Result<()> {
        let response = self.send_request(Request::Ping).await?;
        match response {
            Response::Pong { .. } => {
                self.last_heartbeat = Some(Instant::now());
                Ok(())
            }
            Response::Error { code, message } => {
                anyhow::bail!("Heartbeat failed: {message} (code {code})")
            }
            _ => {
                anyhow::bail!("Unexpected response to heartbeat")
            }
        }
    }

    /// Check if the connection is healthy
    ///
    /// Returns true if connected and last heartbeat was within the health threshold.
    pub fn is_healthy(&self) -> bool {
        if !self.is_connected() {
            return false;
        }

        match self.last_heartbeat {
            Some(last) => last.elapsed() < self.health_threshold,
            None => {
                // If we've never sent a heartbeat but are connected, consider healthy
                // (just connected state)
                self.is_connected()
            }
        }
    }

    /// Add a request to the batch queue
    pub fn batch_add(&mut self, request: Request) {
        self.batch.push(request);
    }

    /// Get the current batch size
    pub fn batch_size(&self) -> usize {
        self.batch.len()
    }

    /// Clear the batch queue
    pub fn batch_clear(&mut self) {
        self.batch.clear();
    }

    /// Execute all batched requests and return responses
    pub async fn batch_execute(&mut self) -> Result<Vec<ResponseEnvelope>> {
        if self.batch.is_empty() {
            return Ok(Vec::new());
        }

        let requests: Vec<Request> = self.batch.drain(..).collect();
        let mut responses = Vec::with_capacity(requests.len());

        for (i, request) in requests.into_iter().enumerate() {
            let id = format!("batch-{i}");
            let envelope = self.send_request_with_id(request, Some(id)).await?;
            responses.push(envelope);
        }

        Ok(responses)
    }

    /// Send a request and return the response
    pub async fn send_request(&mut self, request: Request) -> Result<Response> {
        let envelope = self.send_request_with_id(request, None).await?;
        Ok(envelope.response)
    }

    /// Send a request with a correlation ID and return the response envelope
    pub async fn send_request_with_id(
        &mut self,
        request: Request,
        id: Option<String>,
    ) -> Result<ResponseEnvelope> {
        let (ipc_request, accept_metrics) = match &request {
            Request::Ping => (IpcRequest::Ping, false),
            Request::GetStatus => (IpcRequest::GetMetrics, true),
            Request::Execute {
                command,
                phi_required,
                dry_run,
            } => {
                if *dry_run {
                    (
                        IpcRequest::Validate {
                            command: command.clone(),
                        },
                        false,
                    )
                } else {
                    (
                        IpcRequest::Execute {
                            command: command.clone(),
                            require_phi: Some(*phi_required),
                        },
                        false,
                    )
                }
            }
            Request::GetIntelliSense {
                input, cursor_pos, ..
            } => (
                IpcRequest::GetCompletions {
                    input: input.clone(),
                    cursor_pos: *cursor_pos,
                },
                false,
            ),
            Request::Validate { command, .. } => (
                IpcRequest::Validate {
                    command: command.clone(),
                },
                false,
            ),
            Request::SubscribeMetrics => (IpcRequest::SubscribeMetrics, false),
            Request::UnsubscribeMetrics => (IpcRequest::UnsubscribeMetrics, false),
        };

        let ipc_response = self.send_ipc_request(ipc_request, accept_metrics).await?;
        let response = self.map_ipc_response(&request, ipc_response);

        Ok(ResponseEnvelope {
            id,
            latency_ms: None,
            response,
        })
    }

    async fn send_ipc_request(
        &mut self,
        request: IpcRequest,
        accept_metrics: bool,
    ) -> Result<IpcResponse> {
        // Write phase: borrow writer, send request, then drop the borrow
        {
            let writer = self.writer.as_mut().context("Not connected to service")?;

            let data = rmp_serde::to_vec(&request)?;
            let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();

            writer
                .write_all(&len)
                .await
                .context("Failed to write request length")?;
            writer
                .write_all(&data)
                .await
                .context("Failed to write request payload")?;
            writer.flush().await.context("Failed to flush request")?;
        }

        // Read phase: re-borrow reader each iteration to avoid overlap with update_metrics
        loop {
            let reader = self.reader.as_mut().context("Not connected to service")?;
            let response = tokio::time::timeout(
                self.config.request_timeout,
                Self::read_ipc_response(reader, IPC_MAX_FRAME_BYTES),
            )
            .await
            .context("IPC response timeout")??;
            match response {
                IpcResponse::Metrics(metrics) => {
                    self.update_metrics(metrics.clone());
                    if accept_metrics {
                        return Ok(IpcResponse::Metrics(metrics));
                    }
                }
                other => return Ok(other),
            }
        }
    }

    async fn handshake(&mut self) -> Result<()> {
        let response = self
            .send_ipc_request(
                IpcRequest::Hello {
                    version: IPC_PROTOCOL_VERSION,
                },
                false,
            )
            .await?;

        match response {
            IpcResponse::HelloAck { server_version } if server_version == IPC_PROTOCOL_VERSION => {
                Ok(())
            }
            IpcResponse::Error(message) => anyhow::bail!("Handshake rejected: {message}"),
            other => anyhow::bail!("Unexpected handshake response: {other:?}"),
        }
    }

    async fn read_ipc_response(
        reader: &mut BufReader<OwnedReadHalf>,
        max_size: usize,
    ) -> Result<IpcResponse> {
        let mut len_buf = [0u8; 4];
        reader
            .read_exact(&mut len_buf)
            .await
            .context("Failed to read response length")?;
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 || len > max_size {
            anyhow::bail!("Invalid IPC frame size: {len}");
        }

        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .await
            .context("Failed to read response payload")?;
        let response =
            rmp_serde::from_slice::<IpcResponse>(&buf).context("Failed to decode response")?;
        Ok(response)
    }

    fn map_ipc_response(&self, request: &Request, response: IpcResponse) -> Response {
        match (request, response) {
            (Request::Ping, IpcResponse::Pong) => Response::Pong {
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
            (Request::GetStatus, IpcResponse::Metrics(metrics)) => Response::Status {
                phi: metrics.phi,
                coherence: metrics.coherence,
                consciousness_level: metrics.consciousness_level,
                is_conscious: metrics.is_conscious,
                uptime_secs: metrics.uptime_secs,
            },
            (Request::GetIntelliSense { .. }, IpcResponse::Completions(items)) => {
                let confidence = items
                    .iter()
                    .map(|c| c.similarity)
                    .fold(0.0_f32, |a, b| a.max(b));
                Response::IntelliSenseResult {
                    completions: items,
                    command_preview: None,
                    phi: self.last_metrics.phi,
                    confidence,
                }
            }
            (
                Request::Validate { .. },
                IpcResponse::ValidationResult {
                    valid,
                    safety_level,
                    warnings,
                    ..
                },
            ) => Response::ValidationResult {
                valid,
                safety_level: Self::safety_level_from_string(&safety_level),
                phi_required: 0.0,
                warnings,
            },
            (
                Request::Execute { dry_run: true, .. },
                IpcResponse::ValidationResult {
                    valid,
                    safety_level: _,
                    preview,
                    warnings,
                },
            ) => {
                let output = preview.unwrap_or_else(|| warnings.join("\n"));
                let gate_reason = if valid {
                    Some("dry-run".to_string())
                } else {
                    Some("dry-run failed".to_string())
                };
                Response::ExecutionResult {
                    executed: false,
                    output,
                    phi_at_execution: 0.0,
                    gate_reason,
                }
            }
            (
                Request::Execute { .. },
                IpcResponse::ExecutionResult {
                    success,
                    output,
                    phi_at_execution,
                    vetoed,
                    veto_reason,
                },
            ) => {
                let gate_reason = if vetoed {
                    veto_reason.or(Some("vetoed".to_string()))
                } else {
                    veto_reason
                };
                Response::ExecutionResult {
                    executed: success && !vetoed,
                    output,
                    phi_at_execution,
                    gate_reason,
                }
            }
            (Request::SubscribeMetrics, IpcResponse::Subscribed)
            | (Request::UnsubscribeMetrics, IpcResponse::Subscribed) => Response::Subscribed,
            (_, IpcResponse::HelloAck { server_version }) => Response::Error {
                code: 2,
                message: format!("Unexpected hello ack (server version {server_version})"),
            },
            (_, IpcResponse::Metrics(metrics)) => Response::Status {
                phi: metrics.phi,
                coherence: metrics.coherence,
                consciousness_level: metrics.consciousness_level,
                is_conscious: metrics.is_conscious,
                uptime_secs: metrics.uptime_secs,
            },
            (
                _,
                IpcResponse::ValidationResult {
                    valid,
                    safety_level,
                    warnings,
                    ..
                },
            ) => Response::ValidationResult {
                valid,
                safety_level: Self::safety_level_from_string(&safety_level),
                phi_required: 0.0,
                warnings,
            },
            (_, IpcResponse::Completions(items)) => Response::IntelliSenseResult {
                completions: items,
                command_preview: None,
                phi: self.last_metrics.phi,
                confidence: 0.0,
            },
            (_, IpcResponse::Error(message)) => Response::Error { code: 1, message },
            (_, IpcResponse::Pong) => Response::Pong {
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
            (_, IpcResponse::Subscribed) => Response::Subscribed,
            (
                _,
                IpcResponse::ExecutionResult {
                    success,
                    output,
                    phi_at_execution,
                    vetoed,
                    veto_reason,
                },
            ) => {
                let gate_reason = if vetoed {
                    veto_reason.or(Some("vetoed".to_string()))
                } else {
                    veto_reason
                };
                Response::ExecutionResult {
                    executed: success && !vetoed,
                    output,
                    phi_at_execution,
                    gate_reason,
                }
            }
        }
    }

    fn safety_level_from_string(level: &str) -> SafetyLevelData {
        let lower = level.to_lowercase();
        let color = if lower.contains("destructive")
            || lower.contains("critical")
            || lower.contains("red")
        {
            "red"
        } else if lower.contains("confirm") || lower.contains("yellow") || lower.contains("warn") {
            "yellow"
        } else {
            "green"
        };
        SafetyLevelData {
            level: level.to_string(),
            color: color.to_string(),
            description: level.to_string(),
        }
    }

    /// Get IntelliSense completions for the given input
    pub async fn get_intellisense(
        &mut self,
        input: &str,
        cursor_pos: usize,
        context: &ShellContextData,
    ) -> Result<(Vec<CompletionItem>, Option<String>, f64, f32)> {
        let request = Request::GetIntelliSense {
            input: input.to_string(),
            cursor_pos,
            context: context.clone(),
        };

        let response = self.send_request(request).await?;

        match response {
            Response::IntelliSenseResult {
                completions,
                command_preview,
                phi,
                confidence,
            } => Ok((completions, command_preview, phi, confidence)),
            Response::Error { code, message } => {
                anyhow::bail!("IntelliSense error: {message} (code {code})")
            }
            _ => {
                anyhow::bail!("Unexpected response to IntelliSense request")
            }
        }
    }

    /// Validate a command without executing it
    pub async fn validate_command(
        &mut self,
        command: &str,
        dry_run: bool,
    ) -> Result<(bool, SafetyLevelData, f64, Vec<String>)> {
        let request = Request::Validate {
            command: command.to_string(),
            dry_run,
        };

        let response = self.send_request(request).await?;

        match response {
            Response::ValidationResult {
                valid,
                safety_level,
                phi_required,
                warnings,
            } => Ok((valid, safety_level, phi_required, warnings)),
            Response::Error { code, message } => {
                anyhow::bail!("Validation error: {message} (code {code})")
            }
            _ => {
                anyhow::bail!("Unexpected response to validation request")
            }
        }
    }

    /// Execute a command with Phi gating
    pub async fn execute_gated(
        &mut self,
        command: &str,
        phi_required: f64,
        dry_run: bool,
    ) -> Result<(bool, String, f64, Option<String>)> {
        let request = Request::Execute {
            command: command.to_string(),
            phi_required,
            dry_run,
        };

        let response = self.send_request(request).await?;

        match response {
            Response::ExecutionResult {
                executed,
                output,
                phi_at_execution,
                gate_reason,
            } => Ok((executed, output, phi_at_execution, gate_reason)),
            Response::Error { code, message } => {
                anyhow::bail!("Execution error: {message} (code {code})")
            }
            _ => {
                anyhow::bail!("Unexpected response to execution request")
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[test]
    fn test_connection_state_default() {
        let state = ConnectionState::default();
        assert_eq!(state, ConnectionState::Disconnected);
    }

    #[test]
    fn test_metrics_snapshot_default() {
        let metrics = MetricsSnapshot::default();
        assert_eq!(metrics.phi, 0.5);
        assert!(metrics.is_conscious);
    }

    #[test]
    fn test_ipc_client_creation() {
        let client = ShellIpcClient::with_defaults();
        assert_eq!(client.state(), ConnectionState::Disconnected);
        assert!(!client.is_connected());
    }

    #[test]
    fn test_simulate_metrics() {
        let mut client = ShellIpcClient::with_defaults();
        client.simulate_metrics(0.8, 0.9);
        assert!((client.last_metrics().phi - 0.8).abs() < 0.01);
        assert!(client.last_metrics().is_conscious);
    }

    #[test]
    fn test_ipc_request_serialization() {
        let request = IpcRequest::Execute {
            command: "install nginx".to_string(),
            require_phi: Some(0.5),
        };
        let serialized = rmp_serde::to_vec(&request).unwrap();
        let deserialized: IpcRequest = rmp_serde::from_slice(&serialized).unwrap();
        match deserialized {
            IpcRequest::Execute {
                command,
                require_phi,
            } => {
                assert_eq!(command, "install nginx");
                assert_eq!(require_phi, Some(0.5));
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[tokio::test]
    async fn test_send_ipc_request_hello() {
        let (client_stream, mut server_stream) = match UnixStream::pair() {
            Ok(pair) => pair,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::PermissionDenied {
                    eprintln!("Skipping IPC client test: UnixStream::pair not permitted");
                    return;
                }
                panic!("UnixStream::pair failed: {err}");
            }
        };
        let (read_half, write_half) = client_stream.into_split();

        let mut client = ShellIpcClient::with_defaults();
        client.reader = Some(BufReader::new(read_half));
        client.writer = Some(BufWriter::new(write_half));

        let server_task = tokio::spawn(async move {
            let mut len_buf = [0u8; 4];
            server_stream.read_exact(&mut len_buf).await.unwrap();
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            server_stream.read_exact(&mut buf).await.unwrap();
            let request: IpcRequest = rmp_serde::from_slice(&buf).unwrap();
            assert!(
                matches!(request, IpcRequest::Hello { version } if version == IPC_PROTOCOL_VERSION)
            );

            let response = IpcResponse::HelloAck {
                server_version: IPC_PROTOCOL_VERSION,
            };
            let data = rmp_serde::to_vec(&response).unwrap();
            let len = (data.len() as u32).to_le_bytes();
            server_stream.write_all(&len).await.unwrap();
            server_stream.write_all(&data).await.unwrap();
        });

        let response = match client
            .send_ipc_request(
                IpcRequest::Hello {
                    version: IPC_PROTOCOL_VERSION,
                },
                false,
            )
            .await
        {
            Ok(response) => response,
            Err(err) => {
                let msg = err.to_string();
                if msg.contains("Operation not permitted")
                    || msg.contains("Permission denied")
                    || msg.contains("Failed to flush request")
                {
                    server_task.abort();
                    eprintln!("Skipping IPC client test: {msg}");
                    return;
                }
                panic!("send_ipc_request failed: {err}");
            }
        };
        assert!(
            matches!(response, IpcResponse::HelloAck { server_version } if server_version == IPC_PROTOCOL_VERSION)
        );

        server_task.await.unwrap();
    }
}
