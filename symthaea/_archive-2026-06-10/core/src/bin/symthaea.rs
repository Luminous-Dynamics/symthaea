// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Service Daemon
//!
//! A persistent service that runs the consciousness loop and accepts
//! requests via Unix socket or TCP.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │                 SYMTHAEA SERVICE                      │
//! ├──────────────────────────────────────────────────────┤
//! │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │
//! │  │ Socket     │  │ Request    │  │ Symthaea      │  │
//! │  │ Listener   │─▶│ Handler    │─▶│ Processing     │  │
//! │  └────────────┘  └────────────┘  └────────────────┘  │
//! │                                           │          │
//! │  ┌────────────┐  ┌────────────┐           ▼          │
//! │  │ Background │  │ Response   │◀──────────┘          │
//! │  │ DMN Loop   │  │ + Metrics  │                      │
//! │  └────────────┘  └────────────┘                      │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```bash
//! # Start service (Unix socket)
//! symthaea-service --socket /tmp/symthaea.sock
//!
//! # Start service (TCP)
//! symthaea-service --tcp 127.0.0.1:7777
//!
//! # Client example (netcat)
//! echo '{"type":"query","content":"install nginx"}' | nc -U /tmp/symthaea.sock
//!
//! # Protocol discovery
//! echo '{"type":"protocol"}' | nc 127.0.0.1 7777
//!
//! # Authenticated request envelope
//! echo '{"protocol_version":1,"authorization":"Bearer ...","type":"status"}' | nc 127.0.0.1 7777
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Mutex;
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use wait_timeout::ChildExt;

use symthaea::Symthaea;
use symthaea::control_plane::{
    AuditEvent, AuditLog, MAX_REQUEST_LINE_BYTES, SERVICE_PROTOCOL_VERSION, parse_bearer_token,
    service_known_not_implemented_request_types, service_readonly_programs,
};
use symthaea::hdc::{HDC_DIMENSION, LTC_NEURONS};

// Voice support (feature-gated)
#[cfg(feature = "voice-tts")]
use symthaea::voice::{VoiceConfig, VoiceConversation};

/// Symthaea Service - Consciousness daemon
#[derive(Parser, Debug)]
#[command(name = "symthaea")]
#[command(about = "Symthaea consciousness service with socket interface")]
#[command(version)]
struct Args {
    /// Unix socket path
    #[arg(short, long)]
    socket: Option<PathBuf>,

    /// TCP address (host:port)
    #[arg(short, long)]
    tcp: Option<String>,

    /// Background consciousness loop interval (ms)
    #[arg(long, default_value = "5000")]
    loop_interval: u64,

    /// Auto-sleep interval (seconds, 0 to disable)
    #[arg(long, default_value = "3600")]
    sleep_interval: u64,

    /// State file for persistence
    #[arg(long)]
    state_file: Option<PathBuf>,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Enable voice interface
    #[cfg(feature = "voice-tts")]
    #[arg(long)]
    voice: bool,

    /// Voice input device (default: system default)
    #[cfg(feature = "voice-tts")]
    #[arg(long)]
    voice_input: Option<String>,

    /// Voice ID for TTS (0-9)
    #[cfg(feature = "voice-tts")]
    #[arg(long, default_value = "0")]
    voice_id: u8,
}

/// Shell context for IntelliSense requests
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShellContext {
    /// Current working directory
    #[serde(default)]
    pub cwd: Option<String>,
    /// Recent command history (for context)
    #[serde(default)]
    pub history: Vec<String>,
    /// Environment variables of interest
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

/// Search type for semantic search
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchType {
    Packages,
    Options,
    Services,
}

/// Request from client
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)] // Fields used via serde deserialization
enum Request {
    /// Process a query
    #[serde(rename = "query")]
    Query {
        content: String,
        /// Context for the query (reserved for future use)
        #[serde(default)]
        context: Option<serde_json::Value>,
    },

    /// Get consciousness status
    #[serde(rename = "status")]
    Status,

    /// Trigger introspection
    #[serde(rename = "introspect")]
    Introspect,

    /// Trigger sleep cycle
    #[serde(rename = "sleep")]
    Sleep,

    /// Save state
    #[serde(rename = "save")]
    Save { path: Option<String> },

    /// Shutdown service
    #[serde(rename = "shutdown")]
    Shutdown,

    /// Ping (health check)
    #[serde(rename = "ping")]
    Ping,

    /// Describe the daemon protocol and security envelope
    #[serde(rename = "protocol")]
    Protocol,

    /// Query recent audit events
    #[serde(rename = "audit_events")]
    AuditEvents {
        #[serde(default = "default_audit_limit")]
        limit: usize,
    },

    /// Speak text via TTS
    #[serde(rename = "speak")]
    Speak { text: String },

    /// Listen for speech via STT
    #[serde(rename = "listen")]
    Listen,

    /// Voice conversation turn (listen → process → speak)
    #[serde(rename = "voice_turn")]
    VoiceTurn,

    /// Get voice status
    #[serde(rename = "voice_status")]
    VoiceStatus,

    // ========================================================================
    // SHELL SIDECAR REQUESTS (Phase 1 Protocol Extensions)
    // ========================================================================
    /// IntelliSense completion request
    #[serde(rename = "intellisense")]
    IntelliSense {
        /// Partial input to complete
        partial_input: String,
        /// Cursor position in the input
        #[serde(default)]
        cursor_position: usize,
        /// Shell context for better suggestions
        #[serde(default)]
        context: ShellContext,
    },

    /// Validate a command before execution
    #[serde(rename = "validate_command")]
    ValidateCommand {
        /// Command to validate
        command: String,
        /// If true, perform dry-run analysis
        #[serde(default)]
        dry_run: bool,
    },

    /// Execute command with Phi-gated safety
    #[serde(rename = "execute_gated")]
    ExecuteGated {
        /// Command to execute
        command: String,
        /// Minimum Phi threshold required
        #[serde(default = "default_phi_threshold")]
        phi_threshold: f32,
        /// Require explicit confirmation for destructive commands
        #[serde(default = "default_true")]
        require_confirmation: bool,
    },

    /// Subscribe to real-time consciousness metrics stream
    #[serde(rename = "stream_metrics")]
    StreamMetrics {
        /// Interval between updates in milliseconds
        #[serde(default = "default_metrics_interval")]
        interval_ms: u64,
    },

    // ========================================================================
    // GUI BRIDGE REQUESTS (Phase 4 Protocol Extensions)
    // ========================================================================
    /// GUI widget change notification
    #[serde(rename = "gui_widget_change")]
    GuiWidgetChange {
        /// Widget identifier
        widget_id: String,
        /// New widget value
        new_value: serde_json::Value,
        /// Semantic intent (optional hint)
        #[serde(default)]
        semantic_intent: String,
    },

    /// Parse NixOS configuration for GUI synchronization
    #[serde(rename = "parse_nix_config")]
    ParseNixConfig {
        /// Nix configuration content
        nix_content: String,
        /// Source file path (optional)
        #[serde(default)]
        source_file: Option<String>,
    },

    /// Get partnership/relational consciousness state
    #[serde(rename = "partnership")]
    Partnership,

    /// Semantic search for packages/options/services
    #[serde(rename = "semantic_search")]
    SemanticSearch {
        /// Search query (natural language)
        query: String,
        /// Type of search
        search_type: SearchType,
        /// Maximum results to return
        #[serde(default = "default_search_limit")]
        limit: usize,
    },
}

#[derive(Debug, Deserialize)]
struct WireRequest {
    #[serde(default)]
    protocol_version: Option<u32>,
    #[serde(default)]
    authorization: Option<String>,
    #[serde(flatten)]
    request: Request,
}

// Default value helpers for serde
fn default_phi_threshold() -> f32 {
    0.5
}
fn default_true() -> bool {
    true
}
fn default_metrics_interval() -> u64 {
    1000
}
fn default_search_limit() -> usize {
    10
}
fn default_audit_limit() -> usize {
    50
}

fn request_name(request: &Request) -> &'static str {
    match request {
        Request::Query { .. } => "query",
        Request::Status => "status",
        Request::Introspect => "introspect",
        Request::Sleep => "sleep",
        Request::Save { .. } => "save",
        Request::Shutdown => "shutdown",
        Request::Ping => "ping",
        Request::Protocol => "protocol",
        Request::AuditEvents { .. } => "audit_events",
        Request::Speak { .. } => "speak",
        Request::Listen => "listen",
        Request::VoiceTurn => "voice_turn",
        Request::VoiceStatus => "voice_status",
        Request::IntelliSense { .. } => "intellisense",
        Request::ValidateCommand { .. } => "validate_command",
        Request::ExecuteGated { .. } => "execute_gated",
        Request::StreamMetrics { .. } => "stream_metrics",
        Request::GuiWidgetChange { .. } => "gui_widget_change",
        Request::ParseNixConfig { .. } => "parse_nix_config",
        Request::Partnership => "partnership",
        Request::SemanticSearch { .. } => "semantic_search",
    }
}

/// Response to client
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(clippy::enum_variant_names)]
enum Response {
    /// Query response
    #[serde(rename = "response")]
    QueryResponse {
        content: String,
        confidence: f32,
        safe: bool,
        phi: f32,
        phi_dyad: f64,
        steps_to_emergence: usize,
        processing_time_ms: u64,
    },

    /// Status response
    #[serde(rename = "status")]
    Status {
        uptime_seconds: u64,
        requests_processed: u64,
        consciousness_level: f32,
        memory_count: usize,
        sleep_cycles: u32,
    },

    /// Introspection response
    #[serde(rename = "introspection")]
    Introspection {
        consciousness_level: f32,
        self_loops: usize,
        graph_size: usize,
        complexity: f32,
        short_term_memories: usize,
        long_term_memories: usize,
        // Track 7: Awakening metrics
        phi: f64,
        meta_awareness: f64,
        is_conscious: bool,
        phenomenal_state: String,
        cycles_since_awakening: u64,
        self_model_accuracy: f64,
    },

    /// Sleep response
    #[serde(rename = "sleep_report")]
    SleepReport {
        scaled: usize,
        consolidated: usize,
        pruned: usize,
        patterns_extracted: usize,
    },

    /// Save confirmation
    #[serde(rename = "saved")]
    Saved { path: String },

    /// Shutdown acknowledgment
    #[serde(rename = "shutdown_ack")]
    ShutdownAck,

    /// Pong response
    #[serde(rename = "pong")]
    Pong { timestamp: u64 },

    /// Protocol metadata
    #[serde(rename = "protocol_info")]
    ProtocolInfo {
        protocol_version: u32,
        min_supported_version: u32,
        auth_required: bool,
        auth_scheme: Option<String>,
        execute_gated_mode: String,
        allowed_readonly_programs: Vec<String>,
        known_not_implemented_requests: Vec<String>,
        notes: Vec<String>,
    },

    /// Recent audit events
    #[serde(rename = "audit_events")]
    AuditEvents { events: Vec<AuditEvent> },

    /// Error response
    #[serde(rename = "error")]
    Error { message: String },

    /// Request reached a known but unsupported feature surface
    #[serde(rename = "not_implemented")]
    NotImplemented { feature: String, message: String },

    /// Speech synthesized (TTS complete)
    #[cfg(feature = "voice-tts")]
    #[serde(rename = "spoken")]
    Spoken { text: String, duration_ms: u64 },

    /// Speech transcribed (STT complete)
    #[cfg(feature = "voice-tts")]
    #[serde(rename = "transcribed")]
    Transcribed { text: String, confidence: f32 },

    /// Voice conversation turn complete
    #[cfg(feature = "voice-tts")]
    #[serde(rename = "voice_turn_response")]
    VoiceTurnResponse {
        user_said: String,
        assistant_said: String,
        phi: f32,
        processing_time_ms: u64,
    },

    /// Voice status response
    #[serde(rename = "voice_status")]
    VoiceStatusResponse {
        enabled: bool,
        stt_ready: bool,
        tts_ready: bool,
        voice_id: u8,
    },

    // ========================================================================
    // SHELL SIDECAR RESPONSES (Phase 1 Protocol Extensions)
    // ========================================================================
    /// IntelliSense completion result
    #[serde(rename = "intellisense_result")]
    IntelliSenseResult {
        /// Completion suggestions
        completions: Vec<Completion>,
        /// Multi-step command preview (if applicable)
        command_preview: Option<CommandPreview>,
        /// Current Phi value
        phi: f32,
        /// Confidence in top completion
        confidence: f32,
    },

    /// Command validation result
    #[serde(rename = "validation_result")]
    ValidationResult {
        /// Whether the command is valid
        valid: bool,
        /// Safety classification
        safety_level: SafetyLevel,
        /// Destructiveness classification
        destructiveness: String,
        /// Minimum Phi required for execution
        phi_required: f32,
        /// Warnings about the command
        warnings: Vec<String>,
        /// Suggested alternatives
        suggested_alternatives: Vec<String>,
        /// Rollback hint if available
        rollback_hint: Option<String>,
    },

    /// Gated execution result
    #[serde(rename = "execution_result")]
    ExecutionResult {
        /// Whether the command was executed
        executed: bool,
        /// Command output (if executed)
        output: Option<String>,
        /// Phi at time of execution
        phi_at_execution: f32,
        /// Reason if execution was blocked
        gate_reason: Option<String>,
        /// Whether confirmation is required
        requires_confirmation: bool,
        /// Destructiveness level
        destructiveness: String,
        /// Rollback hint
        rollback_hint: Option<String>,
    },

    /// Real-time metrics update (for streaming)
    #[serde(rename = "metrics_update")]
    MetricsUpdate {
        /// Phi (integrated information)
        phi: f32,
        /// Coherence level
        coherence: f32,
        /// Consciousness level (0-1)
        consciousness_level: f32,
        /// Safety statistics
        safety_checks: u64,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },

    /// Semantic search results
    #[serde(rename = "search_results")]
    SearchResults {
        /// Search results
        results: Vec<SearchResult>,
        /// HDC-based confidence score
        hdc_confidence: f32,
    },

    /// Partnership/relational consciousness state
    #[serde(rename = "partnership")]
    Partnership {
        /// Current relationship stage
        stage: String,
        /// Trust level (0.0-1.0)
        trust: f32,
        /// Vulnerability level (0.0-1.0)
        vulnerability: f32,
        /// Reciprocity level (0.0-1.0)
        reciprocity: f32,
        /// Phi-dyad value
        phi_dyad: f64,
        /// Total interactions
        interactions: u64,
        /// Trajectory points recorded
        trajectory_points: usize,
    },
}

// ============================================================================
// SUPPORTING TYPES FOR SHELL SIDECAR
// ============================================================================

/// Completion suggestion for IntelliSense
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    /// The completion text
    pub text: String,
    /// Description/documentation
    pub description: String,
    /// Completion kind (command, package, option, etc.)
    pub kind: CompletionKind,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Destructiveness level if this is a command
    pub destructiveness: Option<String>,
}

/// Kind of completion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompletionKind {
    Command,
    Package,
    Option,
    Service,
    Path,
    Argument,
}

/// Multi-step command preview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPreview {
    /// Steps that will be executed
    pub steps: Vec<CommandStep>,
    /// Total estimated changes
    pub summary: String,
    /// Overall risk assessment
    pub risk_level: String,
}

/// Single step in command preview
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandStep {
    /// Step number
    pub number: usize,
    /// Description of what this step does
    pub description: String,
    /// Reversible?
    pub reversible: bool,
}

/// Safety level classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SafetyLevel {
    Safe,
    Caution,
    Warning,
    Dangerous,
}

/// Search result for semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Name/identifier
    pub name: String,
    /// Description
    pub description: String,
    /// Result type
    pub result_type: SearchType,
    /// Relevance score (0-1)
    pub relevance: f32,
    /// Attribute path (for packages)
    pub attr_path: Option<String>,
}

/// Service state
struct ServiceState {
    symthaea: Mutex<Symthaea>,
    audit_log: AuditLog,
    start_time: Instant,
    requests_processed: AtomicU64,
    sleep_cycles: AtomicU32,
    auth_enabled: bool,
    state_file: Option<PathBuf>,
    #[cfg(feature = "voice-tts")]
    voice: Mutex<Option<VoiceConversation>>,
    #[cfg(feature = "voice-tts")]
    voice_enabled: bool,
}

#[derive(Clone, Default)]
struct ServiceSecurity {
    bearer_token: Option<String>,
}

fn addr_is_loopback(addr: &str) -> bool {
    if addr.starts_with("localhost:") {
        return true;
    }

    if let Ok(sa) = addr.parse::<std::net::SocketAddr>() {
        return sa.ip().is_loopback();
    }

    false
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn request_requires_auth(request: &Request) -> bool {
    !matches!(request, Request::Ping | Request::Protocol)
}

fn service_auth_error_message() -> String {
    "Authentication required: include top-level field `authorization` with value `Bearer <token>`"
        .to_string()
}

fn run_structured_command(command: &str) -> Result<String, String> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::thread;

    let (program, args) = symthaea::action::parse_command_line(command)
        .map_err(|e| format!("Invalid command: {e}"))?;

    let mut child = Command::new(&program)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn '{program}': {e}"))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Missing stdout pipe".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Missing stderr pipe".to_string())?;

    let stdout_handle = thread::spawn(move || {
        let mut stdout = stdout;
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        buf
    });
    let stderr_handle = thread::spawn(move || {
        let mut stderr = stderr;
        let mut buf = Vec::new();
        let _ = stderr.read_to_end(&mut buf);
        buf
    });

    let timeout = Duration::from_secs(30);
    let status = match child
        .wait_timeout(timeout)
        .map_err(|e| format!("Failed to wait for process: {e}"))?
    {
        Some(status) => status,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            return Err("Command timed out after 30s".to_string());
        }
    };

    let stdout = stdout_handle
        .join()
        .map_err(|_| "stdout reader thread panicked".to_string())?;
    let stderr = stderr_handle
        .join()
        .map_err(|_| "stderr reader thread panicked".to_string())?;

    let stdout = String::from_utf8_lossy(&stdout);
    let stderr = String::from_utf8_lossy(&stderr);
    let mut output = String::new();
    if !stdout.is_empty() {
        output.push_str(&stdout);
    }
    if !stderr.is_empty() {
        if !output.is_empty() && !output.ends_with('\n') {
            output.push('\n');
        }
        output.push_str(&stderr);
    }
    if output.is_empty() {
        output = format!("Command exited with status {:?}", status.code());
    }

    if status.success() {
        Ok(output)
    } else {
        Err(format!(
            "Command exited with status {:?}\n{}",
            status.code(),
            output
        ))
    }
}

async fn write_response<W>(writer: &mut W, response: &Response) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let json = serde_json::to_string(response)?;
    writer.write_all(json.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

struct ProcessedRequestOutcome {
    response: Response,
    shutdown: bool,
}

impl ServiceState {
    async fn new(
        auth_enabled: bool,
        audit_log_path: Option<PathBuf>,
        state_file: Option<PathBuf>,
        #[cfg(feature = "voice-tts")] voice_enabled: bool,
        #[cfg(feature = "voice-tts")] voice_id: u8,
    ) -> Result<Self> {
        // Try to resume from state file if it exists
        let symthaea = if let Some(ref path) = state_file {
            if path.exists() {
                info!("Resuming from state file: {:?}", path);
                let path_str = path.to_string_lossy();
                match Symthaea::resume(&path_str) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("Failed to resume: {}, starting fresh", e);
                        Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await?
                    }
                }
            } else {
                Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await?
            }
        } else {
            Symthaea::new(HDC_DIMENSION, LTC_NEURONS).await?
        };

        // Initialize voice if enabled
        #[cfg(feature = "voice-tts")]
        let voice = if voice_enabled {
            info!("Initializing voice interface (voice_id={})...", voice_id);
            let config = VoiceConfig {
                voice_id,
                ltc_pacing: true,
                ..Default::default()
            };
            match VoiceConversation::new(config) {
                Ok(vc) => {
                    info!("Voice interface ready");
                    Some(vc)
                }
                Err(e) => {
                    warn!("Failed to initialize voice: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            symthaea: Mutex::new(symthaea),
            audit_log: AuditLog::new(audit_log_path, 256),
            start_time: Instant::now(),
            requests_processed: AtomicU64::new(0),
            sleep_cycles: AtomicU32::new(0),
            auth_enabled,
            state_file,
            #[cfg(feature = "voice-tts")]
            voice: Mutex::new(voice),
            #[cfg(feature = "voice-tts")]
            voice_enabled,
        })
    }

    fn record_audit(&self, event: &str, subject: &str, detail: &str) {
        match self.audit_log.record("service", event, subject, detail) {
            Ok(entry) => info!(
                target: "symthaea_service_audit",
                event = %entry.event,
                subject = %entry.subject,
                detail = %entry.detail
            ),
            Err(err) => warn!(
                target: "symthaea_service_audit",
                event = %event,
                subject = %subject,
                detail = %detail,
                error = %err
            ),
        }
    }

    async fn handle_request(&self, request: Request) -> Response {
        self.requests_processed.fetch_add(1, Ordering::Relaxed);

        match request {
            Request::Query {
                content,
                context: _,
            } => {
                let start = Instant::now();
                let mut symthaea = self.symthaea.lock().await;
                match symthaea.process(&content).await {
                    Ok(response) => {
                        let intro = symthaea.introspect();
                        let partnership = symthaea.partnership_state();
                        Response::QueryResponse {
                            content: response.content,
                            confidence: response.confidence,
                            safe: response.safe,
                            phi: intro.consciousness_level,
                            phi_dyad: partnership.phi_dyad,
                            steps_to_emergence: response.steps_to_emergence,
                            processing_time_ms: start.elapsed().as_millis() as u64,
                        }
                    }
                    Err(e) => Response::Error {
                        message: format!("Processing error: {}", e),
                    },
                }
            }

            Request::Status => {
                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };
                Response::Status {
                    uptime_seconds: self.start_time.elapsed().as_secs(),
                    requests_processed: self.requests_processed.load(Ordering::Relaxed),
                    consciousness_level: intro.consciousness_level,
                    memory_count: intro.memory_stats.short_term_count
                        + intro.memory_stats.long_term_count,
                    sleep_cycles: self.sleep_cycles.load(Ordering::Relaxed),
                }
            }

            Request::Introspect => {
                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };
                // Derive consciousness metrics from available data
                let phi = intro.consciousness_level as f64;
                let is_conscious = intro.consciousness_level > 0.5;

                Response::Introspection {
                    consciousness_level: intro.consciousness_level,
                    self_loops: intro.self_loops,
                    graph_size: intro.graph_size,
                    complexity: intro.complexity,
                    short_term_memories: intro.memory_stats.short_term_count,
                    long_term_memories: intro.memory_stats.long_term_count,
                    // Derived awakening metrics
                    phi,
                    meta_awareness: phi * 0.8, // Derived from phi
                    is_conscious,
                    phenomenal_state: if is_conscious {
                        "Aware".to_string()
                    } else {
                        "Dormant".to_string()
                    },
                    cycles_since_awakening: 0, // Not tracked yet
                    self_model_accuracy: intro.complexity as f64 / 10.0,
                }
            }

            Request::Sleep => {
                let mut symthaea = self.symthaea.lock().await;
                match symthaea.sleep().await {
                    Ok(report) => {
                        self.sleep_cycles.fetch_add(1, Ordering::Relaxed);
                        Response::SleepReport {
                            scaled: report.scaled,
                            consolidated: report.consolidated,
                            pruned: report.pruned,
                            patterns_extracted: report.patterns_extracted,
                        }
                    }
                    Err(e) => Response::Error {
                        message: format!("Sleep error: {}", e),
                    },
                }
            }

            Request::Save { path } => {
                let save_path = path
                    .map(PathBuf::from)
                    .or_else(|| self.state_file.clone())
                    .unwrap_or_else(|| PathBuf::from("symthaea-state.bin"));

                let path_str = save_path.to_string_lossy();
                let mut symthaea = self.symthaea.lock().await;
                match symthaea.pause(&path_str) {
                    Ok(()) => Response::Saved {
                        path: save_path.display().to_string(),
                    },
                    Err(e) => Response::Error {
                        message: format!("Save error: {}", e),
                    },
                }
            }

            Request::Shutdown => {
                // Save state before shutdown if configured
                if let Some(ref path) = self.state_file {
                    let path_str = path.to_string_lossy();
                    let mut symthaea = self.symthaea.lock().await;
                    let _ = symthaea.pause(&path_str);
                    info!("State saved to {:?}", path);
                }
                Response::ShutdownAck
            }

            Request::Ping => Response::Pong {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },

            Request::Protocol => Response::ProtocolInfo {
                protocol_version: SERVICE_PROTOCOL_VERSION,
                min_supported_version: SERVICE_PROTOCOL_VERSION,
                auth_required: self.auth_enabled,
                auth_scheme: Some("authorization: \"Bearer <token>\"".to_string()),
                execute_gated_mode: "read_only_allowlist".to_string(),
                allowed_readonly_programs: service_readonly_programs(),
                known_not_implemented_requests: service_known_not_implemented_request_types(),
                notes: {
                    let mut notes = vec![
                        "The daemon accepts a JSON-line envelope with optional protocol_version and authorization fields."
                            .to_string(),
                        "Mutating commands are rejected over the daemon protocol.".to_string(),
                        "gui_widget_change and parse_nix_config are reserved request types that currently return not_implemented."
                            .to_string(),
                    ];
                    if let Some(path) = self.audit_log.file_path() {
                        notes.push(format!(
                            "Audit events are retained in memory and appended to JSONL at {}.",
                            path.display()
                        ));
                    } else {
                        notes.push(
                            "Audit events are retained in memory only unless SYMTHAEA_SERVICE_AUDIT_LOG_PATH is configured."
                                .to_string(),
                        );
                    }
                    notes
                },
            },

            Request::AuditEvents { limit } => Response::AuditEvents {
                events: self.audit_log.list(limit.min(500)),
            },

            // Voice requests
            Request::Speak { text } => {
                #[cfg(feature = "voice-tts")]
                {
                    let mut voice = self.voice.lock().await;
                    if let Some(ref mut voice) = *voice {
                        let start = Instant::now();
                        match voice.speak(&text) {
                            Ok(()) => Response::Spoken {
                                text,
                                duration_ms: start.elapsed().as_millis() as u64,
                            },
                            Err(e) => Response::Error {
                                message: format!("TTS error: {}", e),
                            },
                        }
                    } else {
                        Response::Error {
                            message: "Voice not enabled".into(),
                        }
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    let _ = text;
                    Response::Error {
                        message: "Voice feature not compiled (build with --features voice-tts)"
                            .into(),
                    }
                }
            }

            Request::Listen => {
                #[cfg(feature = "voice-tts")]
                {
                    let mut voice = self.voice.lock().await;
                    if let Some(ref mut voice) = *voice {
                        match voice.listen() {
                            Ok(text) => Response::Transcribed {
                                text,
                                confidence: 0.9,
                            },
                            Err(e) => Response::Error {
                                message: format!("STT error: {}", e),
                            },
                        }
                    } else {
                        Response::Error {
                            message: "Voice not enabled".into(),
                        }
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    Response::Error {
                        message: "Voice feature not compiled (build with --features voice-tts)"
                            .into(),
                    }
                }
            }

            Request::VoiceTurn => {
                #[cfg(feature = "voice-tts")]
                {
                    let user_said = {
                        let mut voice = self.voice.lock().await;
                        if let Some(ref mut voice) = *voice {
                            match voice.listen() {
                                Ok(text) => text,
                                Err(e) => {
                                    return Response::Error {
                                        message: format!("Listen error: {}", e),
                                    };
                                }
                            }
                        } else {
                            return Response::Error {
                                message: "Voice not enabled".into(),
                            };
                        }
                    };

                    let start = Instant::now();
                    let (assistant_said, phi) = {
                        let mut symthaea = self.symthaea.lock().await;
                        match symthaea.process(&user_said).await {
                            Ok(response) => {
                                let intro = symthaea.introspect();
                                (response.content, intro.consciousness_level)
                            }
                            Err(e) => {
                                return Response::Error {
                                    message: format!("Processing error: {}", e),
                                };
                            }
                        }
                    };

                    {
                        let mut voice = self.voice.lock().await;
                        if let Some(ref mut voice) = *voice {
                            if let Err(e) = voice.speak(&assistant_said) {
                                warn!("TTS error (continuing): {}", e);
                            }
                        }
                    }

                    Response::VoiceTurnResponse {
                        user_said,
                        assistant_said,
                        phi,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    Response::Error {
                        message: "Voice feature not compiled (build with --features voice-tts)"
                            .into(),
                    }
                }
            }

            Request::VoiceStatus => {
                #[cfg(feature = "voice-tts")]
                {
                    let voice = self.voice.lock().await;
                    Response::VoiceStatusResponse {
                        enabled: self.voice_enabled,
                        stt_ready: voice.is_some(),
                        tts_ready: voice.is_some(),
                        voice_id: voice.as_ref().map(|_| 0).unwrap_or(0),
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    Response::VoiceStatusResponse {
                        enabled: false,
                        stt_ready: false,
                        tts_ready: false,
                        voice_id: 0,
                    }
                }
            }

            Request::Partnership => {
                let state = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.partnership_state()
                };
                Response::Partnership {
                    stage: format!("{:?}", state.stage),
                    trust: state.trust,
                    vulnerability: state.vulnerability,
                    reciprocity: state.reciprocity,
                    phi_dyad: state.phi_dyad,
                    interactions: state.interactions,
                    trajectory_points: state.trajectory_points,
                }
            }

            // ================================================================
            // SHELL SIDECAR HANDLERS (Phase 1)
            // ================================================================
            Request::IntelliSense {
                partial_input,
                cursor_position: _,
                context: _,
            } => {
                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };

                // Generate completions based on partial input
                // For now, provide basic NixOS-aware completions
                let completions = generate_intellisense_completions(&partial_input);

                // Generate command preview if input looks like a complete command
                let command_preview = if partial_input.contains(' ') {
                    generate_command_preview(&partial_input)
                } else {
                    None
                };

                let confidence = completions.first().map(|c| c.confidence).unwrap_or(0.0);

                Response::IntelliSenseResult {
                    completions,
                    command_preview,
                    phi: intro.consciousness_level, // Use consciousness_level as phi
                    confidence,
                }
            }

            Request::ValidateCommand {
                command,
                dry_run: _,
            } => {
                use symthaea::action::{
                    DestructivenessLevel, RemoteCommandCapability,
                    classify_command_destructiveness, classify_remote_command_capability,
                    get_rollback_hint, parse_command_line,
                };

                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };

                let (program, args, parse_error) = match parse_command_line(&command) {
                    Ok((program, args)) => (program, args, None),
                    Err(err) => ("".to_string(), Vec::new(), Some(err)),
                };
                let capability = if parse_error.is_none() {
                    classify_remote_command_capability(&program, &args)
                } else {
                    Err("command could not be parsed".to_string())
                };
                let read_only_capable =
                    matches!(&capability, Ok(RemoteCommandCapability::ReadOnly));
                let mutating_capable = matches!(&capability, Ok(RemoteCommandCapability::Mutating));

                // Classify destructiveness
                let destructiveness = classify_command_destructiveness(&program, &args);
                let rollback_hint = get_rollback_hint(&program, &args);

                // Determine safety level
                let safety_level = match destructiveness {
                    DestructivenessLevel::ReadOnly => SafetyLevel::Safe,
                    DestructivenessLevel::Reversible => SafetyLevel::Caution,
                    DestructivenessLevel::NeedsConfirmation => SafetyLevel::Warning,
                    DestructivenessLevel::Destructive => SafetyLevel::Dangerous,
                };

                // Calculate required Phi based on destructiveness
                let phi_required = match destructiveness {
                    DestructivenessLevel::ReadOnly => 0.0,
                    DestructivenessLevel::Reversible => 0.3,
                    DestructivenessLevel::NeedsConfirmation => 0.5,
                    DestructivenessLevel::Destructive => 0.7,
                };

                // Generate warnings
                let current_phi = intro.consciousness_level as f64;
                let mut warnings = Vec::new();
                if let Some(err) = &parse_error {
                    warnings.push(err.clone());
                }
                if let Err(err) = &capability {
                    warnings.push(err.clone());
                }
                if mutating_capable {
                    warnings.push(
                        "Mutating commands are not executable over the daemon protocol".to_string(),
                    );
                }
                if destructiveness.requires_confirmation() {
                    warnings.push(format!(
                        "This command requires confirmation: {}",
                        destructiveness.description()
                    ));
                }
                if current_phi < phi_required as f64 {
                    warnings.push(format!(
                        "Current Phi ({:.2}) is below required threshold ({:.2})",
                        current_phi, phi_required
                    ));
                }

                Response::ValidationResult {
                    valid: read_only_capable,
                    safety_level,
                    destructiveness: format!("{:?}", destructiveness),
                    phi_required,
                    warnings,
                    suggested_alternatives: Vec::new(),
                    rollback_hint,
                }
            }

            Request::ExecuteGated {
                command,
                phi_threshold,
                require_confirmation,
            } => {
                use symthaea::action::{
                    RemoteCommandCapability, classify_command_destructiveness,
                    classify_remote_command_capability, get_rollback_hint, parse_command_line,
                };

                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };
                let current_phi = intro.consciousness_level; // Use consciousness_level as phi

                // Parse command
                let (program, args) = match parse_command_line(&command) {
                    Ok(parsed) => parsed,
                    Err(err) => {
                        return Response::ExecutionResult {
                            executed: false,
                            output: None,
                            phi_at_execution: current_phi,
                            gate_reason: Some(err),
                            requires_confirmation: false,
                            destructiveness: "Invalid".to_string(),
                            rollback_hint: None,
                        };
                    }
                };

                let destructiveness = classify_command_destructiveness(&program, &args);
                let rollback_hint = get_rollback_hint(&program, &args);
                let capability = match classify_remote_command_capability(&program, &args) {
                    Ok(capability) => capability,
                    Err(err) => {
                        self.record_audit("command_blocked", "execute_gated", &err);
                        return Response::ExecutionResult {
                            executed: false,
                            output: None,
                            phi_at_execution: current_phi,
                            gate_reason: Some(err),
                            requires_confirmation: false,
                            destructiveness: format!("{:?}", destructiveness),
                            rollback_hint,
                        };
                    }
                };
                if capability != RemoteCommandCapability::ReadOnly {
                    let reason =
                        "Mutating commands are not permitted over the daemon protocol".to_string();
                    self.record_audit("command_blocked", "execute_gated", &reason);
                    return Response::ExecutionResult {
                        executed: false,
                        output: None,
                        phi_at_execution: current_phi,
                        gate_reason: Some(reason),
                        requires_confirmation: false,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    };
                }
                let needs_confirmation =
                    require_confirmation && destructiveness.requires_confirmation();

                // Check Phi gate
                if current_phi < phi_threshold {
                    return Response::ExecutionResult {
                        executed: false,
                        output: None,
                        phi_at_execution: current_phi,
                        gate_reason: Some(format!(
                            "Phi {:.2} is below threshold {:.2}. Center yourself before executing.",
                            current_phi, phi_threshold
                        )),
                        requires_confirmation: needs_confirmation,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    };
                }

                // If confirmation required and command is destructive, don't execute yet
                if needs_confirmation {
                    return Response::ExecutionResult {
                        executed: false,
                        output: None,
                        phi_at_execution: current_phi,
                        gate_reason: Some(format!(
                            "Confirmation required for {:?} command",
                            destructiveness
                        )),
                        requires_confirmation: true,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    };
                }

                self.record_audit("command_execute_attempt", "execute_gated", &command);
                match run_structured_command(&command) {
                    Ok(output) => Response::ExecutionResult {
                        executed: true,
                        output: Some(output),
                        phi_at_execution: current_phi,
                        gate_reason: None,
                        requires_confirmation: false,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    },
                    Err(e) => {
                        self.record_audit("command_execute_failed", "execute_gated", &e);
                        Response::ExecutionResult {
                            executed: false,
                            output: None,
                            phi_at_execution: current_phi,
                            gate_reason: Some(e),
                            requires_confirmation: false,
                            destructiveness: format!("{:?}", destructiveness),
                            rollback_hint,
                        }
                    }
                }
            }

            Request::StreamMetrics { interval_ms: _ } => {
                // For now, return a single metrics snapshot
                // Full streaming would require a different connection model
                let intro = {
                    let symthaea = self.symthaea.lock().await;
                    symthaea.introspect()
                };

                Response::MetricsUpdate {
                    phi: intro.consciousness_level, // Use consciousness_level as phi approximation
                    coherence: intro.consciousness_level,
                    consciousness_level: intro.consciousness_level,
                    safety_checks: self.requests_processed.load(Ordering::Relaxed),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                }
            }

            // ================================================================
            // GUI BRIDGE HANDLERS (Phase 4 - currently not implemented)
            // ================================================================
            Request::GuiWidgetChange { .. } => {
                self.record_audit(
                    "not_implemented",
                    "gui_widget_change",
                    "GUI bridge handlers are not implemented in the service daemon",
                );
                Response::NotImplemented {
                    feature: "gui_widget_change".to_string(),
                    message: "GUI bridge handlers are not implemented in this daemon build"
                        .to_string(),
                }
            }

            Request::ParseNixConfig { .. } => {
                self.record_audit(
                    "not_implemented",
                    "parse_nix_config",
                    "Nix GUI synchronization parsing is not implemented in the service daemon",
                );
                Response::NotImplemented {
                    feature: "parse_nix_config".to_string(),
                    message:
                        "Nix GUI synchronization parsing is not implemented in this daemon build"
                            .to_string(),
                }
            }

            Request::SemanticSearch {
                query,
                search_type,
                limit,
            } => {
                // Basic search implementation using consciousness
                let results = generate_search_results(&query, search_type, limit);

                Response::SearchResults {
                    results,
                    hdc_confidence: 0.75, // Placeholder until HDC integration
                }
            }
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS FOR SHELL SIDECAR
// ============================================================================

/// Generate IntelliSense completions for partial input
fn generate_intellisense_completions(partial: &str) -> Vec<Completion> {
    use symthaea::action::classify_command_destructiveness;

    let partial_lower = partial.to_lowercase();
    let mut completions = Vec::new();

    // NixOS command completions
    let nix_commands = [
        (
            "nix search nixpkgs",
            "Search for packages in nixpkgs",
            CompletionKind::Command,
        ),
        (
            "nix-env -i",
            "Install a package to user profile",
            CompletionKind::Command,
        ),
        (
            "nix-env -q",
            "Query installed packages",
            CompletionKind::Command,
        ),
        (
            "nix flake show",
            "Show flake outputs",
            CompletionKind::Command,
        ),
        (
            "nixos-rebuild switch",
            "Rebuild and switch to new configuration",
            CompletionKind::Command,
        ),
        (
            "nixos-rebuild test",
            "Build and test configuration without switching",
            CompletionKind::Command,
        ),
        (
            "nixos-rebuild dry-run",
            "Show what would be built",
            CompletionKind::Command,
        ),
        (
            "nix-collect-garbage -d",
            "Delete old generations (destructive)",
            CompletionKind::Command,
        ),
        (
            "systemctl status",
            "Show service status",
            CompletionKind::Command,
        ),
        (
            "systemctl restart",
            "Restart a service",
            CompletionKind::Command,
        ),
        (
            "journalctl -u",
            "View service logs",
            CompletionKind::Command,
        ),
    ];

    for (cmd, desc, kind) in &nix_commands {
        if cmd.to_lowercase().starts_with(&partial_lower) || partial_lower.is_empty() {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            let (prog, args) = if parts.is_empty() {
                ("", vec![])
            } else {
                (parts[0], parts[1..].iter().map(|s| s.to_string()).collect())
            };

            let destructiveness = classify_command_destructiveness(prog, &args);

            completions.push(Completion {
                text: cmd.to_string(),
                description: desc.to_string(),
                kind: *kind,
                confidence: calculate_completion_confidence(&partial_lower, cmd),
                destructiveness: Some(format!("{:?}", destructiveness)),
            });
        }
    }

    // Sort by confidence descending
    completions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    completions.truncate(10);

    completions
}

/// Calculate completion confidence based on prefix match
fn calculate_completion_confidence(partial: &str, full: &str) -> f32 {
    if partial.is_empty() {
        return 0.5;
    }

    let full_lower = full.to_lowercase();
    if full_lower.starts_with(partial) {
        let ratio = partial.len() as f32 / full.len() as f32;
        0.7 + (ratio * 0.3)
    } else if full_lower.contains(partial) {
        0.5
    } else {
        0.3
    }
}

/// Generate command preview for multi-step operations
fn generate_command_preview(command: &str) -> Option<CommandPreview> {
    use symthaea::action::{DestructivenessLevel, classify_command_destructiveness};

    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    let (prog, args): (&str, Vec<String>) =
        (parts[0], parts[1..].iter().map(|s| s.to_string()).collect());

    let destructiveness = classify_command_destructiveness(prog, &args);

    // Generate steps based on command type
    let steps = if command.contains("nixos-rebuild") {
        vec![
            CommandStep {
                number: 1,
                description: "Evaluate configuration".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 2,
                description: "Build system derivation".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 3,
                description: "Activate new generation".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 4,
                description: "Update boot loader".to_string(),
                reversible: true,
            },
        ]
    } else if command.contains("nix-env -i") {
        vec![
            CommandStep {
                number: 1,
                description: "Resolve package".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 2,
                description: "Download/build package".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 3,
                description: "Install to profile".to_string(),
                reversible: true,
            },
        ]
    } else if command.contains("nix-collect-garbage") {
        vec![
            CommandStep {
                number: 1,
                description: "Find unused store paths".to_string(),
                reversible: true,
            },
            CommandStep {
                number: 2,
                description: "Delete old generations".to_string(),
                reversible: false,
            },
            CommandStep {
                number: 3,
                description: "Remove store paths".to_string(),
                reversible: false,
            },
        ]
    } else {
        return None;
    };

    let risk_level = match destructiveness {
        DestructivenessLevel::ReadOnly => "LOW",
        DestructivenessLevel::Reversible => "MEDIUM",
        DestructivenessLevel::NeedsConfirmation => "HIGH",
        DestructivenessLevel::Destructive => "CRITICAL",
    };

    Some(CommandPreview {
        steps,
        summary: format!("{} ({:?})", command, destructiveness),
        risk_level: risk_level.to_string(),
    })
}

/// Generate search results for semantic search
fn generate_search_results(
    query: &str,
    search_type: SearchType,
    limit: usize,
) -> Vec<SearchResult> {
    let query_lower = query.to_lowercase();
    let mut results = Vec::new();

    match search_type {
        SearchType::Packages => {
            // Common NixOS packages
            let packages = [
                ("firefox", "Web browser", "pkgs.firefox"),
                ("vim", "Text editor", "pkgs.vim"),
                ("neovim", "Modern vim fork", "pkgs.neovim"),
                ("git", "Version control", "pkgs.git"),
                ("rustc", "Rust compiler", "pkgs.rustc"),
                ("python3", "Python interpreter", "pkgs.python3"),
                ("nginx", "Web server", "pkgs.nginx"),
                ("docker", "Container runtime", "pkgs.docker"),
                ("htop", "Process viewer", "pkgs.htop"),
                ("tmux", "Terminal multiplexer", "pkgs.tmux"),
            ];

            for (name, desc, attr) in &packages {
                if name.contains(&query_lower) || desc.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        name: name.to_string(),
                        description: desc.to_string(),
                        result_type: SearchType::Packages,
                        relevance: if name.starts_with(&query_lower) {
                            0.9
                        } else {
                            0.7
                        },
                        attr_path: Some(attr.to_string()),
                    });
                }
            }
        }
        SearchType::Options => {
            let options = [
                ("services.nginx.enable", "Enable nginx web server"),
                ("services.openssh.enable", "Enable SSH server"),
                ("networking.firewall.enable", "Enable firewall"),
                ("boot.loader.systemd-boot.enable", "Use systemd-boot"),
                ("users.users", "User account configuration"),
            ];

            for (name, desc) in &options {
                if name.contains(&query_lower) || desc.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        name: name.to_string(),
                        description: desc.to_string(),
                        result_type: SearchType::Options,
                        relevance: 0.8,
                        attr_path: None,
                    });
                }
            }
        }
        SearchType::Services => {
            let services = [
                ("nginx", "High-performance web server"),
                ("postgresql", "Advanced SQL database"),
                ("redis", "In-memory data store"),
                ("docker", "Container service"),
                ("sshd", "SSH daemon"),
            ];

            for (name, desc) in &services {
                if name.contains(&query_lower) || desc.to_lowercase().contains(&query_lower) {
                    results.push(SearchResult {
                        name: name.to_string(),
                        description: desc.to_string(),
                        result_type: SearchType::Services,
                        relevance: 0.85,
                        attr_path: None,
                    });
                }
            }
        }
    }

    results.truncate(limit);
    results
}

async fn process_request_line(
    line: &str,
    state: Arc<ServiceState>,
    security: ServiceSecurity,
) -> Result<Option<ProcessedRequestOutcome>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }

    if line.as_bytes().len() > MAX_REQUEST_LINE_BYTES {
        state.record_audit(
            "request_too_large",
            "unknown",
            "request line exceeded service limit",
        );
        return Ok(Some(ProcessedRequestOutcome {
            response: Response::Error {
                message: format!(
                    "Request exceeds maximum line length of {} bytes",
                    MAX_REQUEST_LINE_BYTES
                ),
            },
            shutdown: false,
        }));
    }

    debug!("Received: {}", line);

    let wire_request: WireRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            state.record_audit("invalid_json", "unknown", &e.to_string());
            return Ok(Some(ProcessedRequestOutcome {
                response: Response::Error {
                    message: format!("Invalid JSON: {}", e),
                },
                shutdown: false,
            }));
        }
    };

    if let Some(version) = wire_request.protocol_version {
        if version != SERVICE_PROTOCOL_VERSION {
            state.record_audit(
                "protocol_version_rejected",
                request_name(&wire_request.request),
                &format!(
                    "client sent protocol_version={}, supported={}",
                    version, SERVICE_PROTOCOL_VERSION
                ),
            );
            return Ok(Some(ProcessedRequestOutcome {
                response: Response::Error {
                    message: format!(
                        "Unsupported protocol_version {} (supported: {})",
                        version, SERVICE_PROTOCOL_VERSION
                    ),
                },
                shutdown: false,
            }));
        }
    }

    if request_requires_auth(&wire_request.request) {
        if let Some(expected_token) = security.bearer_token.as_deref() {
            let provided_token = wire_request
                .authorization
                .as_deref()
                .and_then(parse_bearer_token);
            if provided_token != Some(expected_token) {
                state.record_audit(
                    "auth_failed",
                    request_name(&wire_request.request),
                    "missing or invalid bearer token",
                );
                return Ok(Some(ProcessedRequestOutcome {
                    response: Response::Error {
                        message: service_auth_error_message(),
                    },
                    shutdown: false,
                }));
            }
        }
    }

    let request = wire_request.request;
    state.record_audit(
        "request_received",
        request_name(&request),
        "request accepted",
    );

    let shutdown = matches!(request, Request::Shutdown);
    let response = state.handle_request(request).await;

    if shutdown {
        state.record_audit(
            "shutdown_requested",
            "shutdown",
            "authenticated shutdown accepted",
        );
    }

    Ok(Some(ProcessedRequestOutcome { response, shutdown }))
}

/// Handle a single connection
async fn handle_connection<S>(
    mut stream: S,
    state: Arc<ServiceState>,
    security: ServiceSecurity,
) -> Result<bool>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (reader, mut writer) = tokio::io::split(&mut stream);
    let mut reader = BufReader::new(reader);
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).await?;

        if bytes_read == 0 {
            break;
        }

        let Some(outcome) = process_request_line(&line, state.clone(), security.clone()).await?
        else {
            continue;
        };

        write_response(&mut writer, &outcome.response).await?;

        if outcome.shutdown {
            return Ok(true);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod protocol_tests {
    use super::*;
    use serde_json::json;

    fn service_request_type_names() -> Vec<&'static str> {
        vec![
            "ping",
            "protocol",
            "status",
            "introspect",
            "sleep",
            "save",
            "shutdown",
            "audit_events",
            "query",
            "speak",
            "listen",
            "voice_turn",
            "voice_status",
            "intellisense",
            "validate_command",
            "execute_gated",
            "stream_metrics",
            "gui_widget_change",
            "parse_nix_config",
            "semantic_search",
            "partnership",
        ]
    }

    async fn test_state(auth_enabled: bool) -> Arc<ServiceState> {
        Arc::new(
            ServiceState::new(
                auth_enabled,
                None,
                None,
                #[cfg(feature = "voice-tts")]
                false,
                #[cfg(feature = "voice-tts")]
                0,
            )
            .await
            .expect("service state"),
        )
    }

    fn test_security(token: Option<&str>) -> ServiceSecurity {
        ServiceSecurity {
            bearer_token: token.map(str::to_string),
        }
    }

    async fn process_json(
        value: serde_json::Value,
        state: Arc<ServiceState>,
        security: ServiceSecurity,
    ) -> Response {
        let line = serde_json::to_string(&value).expect("request JSON");
        process_request_line(&line, state, security)
            .await
            .expect("request outcome")
            .expect("non-empty request")
            .response
    }

    fn response_json(response: &Response) -> serde_json::Value {
        serde_json::to_value(response).expect("response JSON")
    }

    #[tokio::test]
    async fn protocol_request_reports_version_and_auth_scheme() {
        let state = test_state(true).await;
        let response = process_json(
            json!({"type": "protocol"}),
            state,
            test_security(Some("secret-token")),
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "protocol_info");
        assert_eq!(json["protocol_version"], SERVICE_PROTOCOL_VERSION);
        assert_eq!(json["auth_required"], true);
        assert!(
            json["auth_scheme"]
                .as_str()
                .expect("auth scheme")
                .contains("Bearer")
        );
        assert_eq!(
            json["allowed_readonly_programs"]
                .as_array()
                .expect("allowed_readonly_programs")
                .len(),
            service_readonly_programs().len()
        );
        assert_eq!(
            json["known_not_implemented_requests"]
                .as_array()
                .expect("known_not_implemented_requests")
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>(),
            service_known_not_implemented_request_types()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert!(json["notes"].as_array().expect("notes").iter().any(|note| {
            note.as_str()
                .unwrap_or_default()
                .contains("SYMTHAEA_SERVICE_AUDIT_LOG_PATH")
        }));
    }

    #[test]
    fn protocol_schema_covers_runtime_request_types() {
        let schema: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/api/service-protocol-v1.schema.json"
        )))
        .expect("schema JSON");

        let request_types = schema["properties"]["type"]["enum"]
            .as_array()
            .expect("type enum")
            .iter()
            .filter_map(|value| value.as_str())
            .collect::<Vec<_>>();

        for request_type in service_request_type_names() {
            assert!(
                request_types.contains(&request_type),
                "schema is missing runtime request type {request_type}"
            );
        }
    }

    #[tokio::test]
    async fn protected_request_requires_bearer_auth() {
        let state = test_state(true).await;
        let response = process_json(
            json!({"type": "status"}),
            state,
            test_security(Some("secret-token")),
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "error");
        assert!(
            json["message"]
                .as_str()
                .expect("message")
                .contains("Authentication required")
        );
    }

    #[tokio::test]
    async fn audit_events_return_recorded_entries() {
        let state = test_state(true).await;
        let security = test_security(Some("secret-token"));

        let _ = process_json(
            json!({
                "authorization": "Bearer secret-token",
                "type": "status"
            }),
            state.clone(),
            security.clone(),
        )
        .await;

        let response = process_json(
            json!({
                "authorization": "Bearer secret-token",
                "type": "audit_events",
                "limit": 5
            }),
            state,
            security,
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "audit_events");
        let events = json["events"].as_array().expect("events");
        assert!(!events.is_empty(), "expected at least one audit event");
        assert!(
            events
                .iter()
                .any(|event| event["event"] == "request_received" && event["subject"] == "status")
        );
    }

    #[tokio::test]
    async fn protocol_version_mismatch_is_rejected() {
        let state = test_state(false).await;
        let response = process_json(
            json!({
                "protocol_version": SERVICE_PROTOCOL_VERSION + 1,
                "type": "ping"
            }),
            state,
            test_security(None),
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "error");
        assert!(
            json["message"]
                .as_str()
                .expect("message")
                .contains("Unsupported protocol_version")
        );
    }

    #[tokio::test]
    async fn oversized_request_is_rejected() {
        let state = test_state(false).await;
        let line = format!(
            "{{\"type\":\"ping\",\"padding\":\"{}\"}}",
            "a".repeat(MAX_REQUEST_LINE_BYTES)
        );

        let response = process_request_line(&line, state, test_security(None))
            .await
            .expect("request outcome")
            .expect("response")
            .response;

        let json = response_json(&response);
        assert_eq!(json["type"], "error");
        assert!(
            json["message"]
                .as_str()
                .expect("message")
                .contains("maximum line length")
        );
    }

    #[tokio::test]
    async fn execute_gated_rejects_mutating_command() {
        let state = test_state(true).await;
        let response = process_json(
            json!({
                "authorization": "Bearer secret-token",
                "type": "execute_gated",
                "command": "touch /tmp/symthaea-should-not-exist"
            }),
            state,
            test_security(Some("secret-token")),
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "execution_result");
        assert_eq!(json["executed"], false);
        assert!(
            json["gate_reason"]
                .as_str()
                .expect("gate_reason")
                .contains("Mutating commands")
        );
    }

    #[tokio::test]
    async fn gui_bridge_requests_return_not_implemented() {
        let state = test_state(false).await;
        let response = process_json(
            json!({
                "type": "gui_widget_change",
                "widget_id": "sidebar",
                "new_value": true,
                "semantic_intent": "toggle"
            }),
            state,
            test_security(None),
        )
        .await;

        let json = response_json(&response);
        assert_eq!(json["type"], "not_implemented");
        assert_eq!(json["feature"], "gui_widget_change");
    }
}

/// Background consciousness loop
async fn consciousness_loop(state: Arc<ServiceState>, interval_ms: u64, sleep_interval: u64) {
    let mut ticker = interval(Duration::from_millis(interval_ms));
    let mut sleep_counter = 0u64;

    loop {
        ticker.tick().await;

        // Simple consciousness maintenance
        {
            let symthaea = state.symthaea.lock().await;
            let intro = symthaea.introspect();
            // Derive metrics from available data
            let phi = intro.consciousness_level as f64;
            let is_conscious = intro.consciousness_level > 0.5;
            debug!(
                "Consciousness loop: level={:.2}% | Φ={:.3} | self_loops={} | conscious={}",
                intro.consciousness_level * 100.0,
                phi,
                intro.self_loops,
                is_conscious
            );
        }

        // Auto-sleep check
        if sleep_interval > 0 {
            sleep_counter += interval_ms;
            if sleep_counter >= sleep_interval * 1000 {
                sleep_counter = 0;
                info!("Triggering automatic sleep cycle");
                let mut symthaea = state.symthaea.lock().await;
                if let Ok(report) = symthaea.sleep().await {
                    state.sleep_cycles.fetch_add(1, Ordering::Relaxed);
                    info!(
                        "Sleep complete: consolidated={}, pruned={}",
                        report.consolidated, report.pruned
                    );
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let filter = if args.verbose {
        "symthaea=debug,symthaea_service=debug"
    } else {
        "symthaea=info,symthaea_service=info"
    };
    tracing_subscriber::fmt().with_env_filter(filter).init();

    // Validate arguments
    if args.socket.is_none() && args.tcp.is_none() {
        anyhow::bail!("Must specify either --socket or --tcp");
    }

    let service_bearer_token = std::env::var("SYMTHAEA_SERVICE_BEARER_TOKEN")
        .ok()
        .and_then(|t| if t.trim().is_empty() { None } else { Some(t) });
    let service_audit_log_path = std::env::var("SYMTHAEA_SERVICE_AUDIT_LOG_PATH")
        .ok()
        .and_then(|p| {
            if p.trim().is_empty() {
                None
            } else {
                Some(PathBuf::from(p))
            }
        });
    let insecure_allow_unauth = env_truthy("SYMTHAEA_SERVICE_INSECURE_ALLOW_UNAUTH");

    if let Some(path) = service_audit_log_path.as_ref() {
        info!(
            "Service audit log persistence enabled at {}",
            path.display()
        );
    } else {
        warn!(
            "Service audit events are retained in memory only; set SYMTHAEA_SERVICE_AUDIT_LOG_PATH for JSONL persistence"
        );
    }

    if let Some(ref tcp_addr) = args.tcp {
        if !addr_is_loopback(tcp_addr) && service_bearer_token.is_none() && !insecure_allow_unauth {
            eprintln!("Refusing to bind Symthaea service to non-loopback address without auth.");
            eprintln!("  addr: {}", tcp_addr);
            eprintln!();
            eprintln!("Set one of:");
            eprintln!("  - SYMTHAEA_SERVICE_BEARER_TOKEN=...   (recommended)");
            eprintln!("  - SYMTHAEA_SERVICE_INSECURE_ALLOW_UNAUTH=1   (NOT recommended)");
            std::process::exit(2);
        }

        if !addr_is_loopback(tcp_addr) && service_bearer_token.is_none() && insecure_allow_unauth {
            eprintln!("WARNING: Symthaea service is binding publicly without auth (insecure).");
            eprintln!("  addr: {}", tcp_addr);
        }
    }

    println!("\n🌟 Symthaea Service Starting...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Initialize state
    info!("Initializing consciousness...");
    let state = Arc::new(
        ServiceState::new(
            service_bearer_token.is_some(),
            service_audit_log_path,
            args.state_file.clone(),
            #[cfg(feature = "voice-tts")]
            args.voice,
            #[cfg(feature = "voice-tts")]
            args.voice_id,
        )
        .await
        .context("Failed to initialize service state")?,
    );
    let security = ServiceSecurity {
        bearer_token: service_bearer_token,
    };

    {
        let intro = {
            let symthaea = state.symthaea.lock().await;
            symthaea.introspect()
        };
        // Derive consciousness metrics
        let phi = intro.consciousness_level as f64;
        let is_conscious = intro.consciousness_level > 0.5;

        println!("✅ Consciousness initialized:");
        println!("   • HDC Dimension: {}", HDC_DIMENSION);
        println!("   • LTC Neurons: {}", LTC_NEURONS);
        println!(
            "   • Consciousness Level: {:.1}%",
            intro.consciousness_level * 100.0
        );
        println!("   • Graph Size: {} states", intro.graph_size);
        println!("   • Self-Loops: {}", intro.self_loops);
        println!("   • λ₂ (Spectral Connectivity): {:.3}", phi);
        println!(
            "   • Is Conscious: {}",
            if is_conscious {
                "✅ Yes"
            } else {
                "🔄 Awakening..."
            }
        );

        #[cfg(feature = "voice-tts")]
        {
            if state.voice_enabled {
                let voice = state.voice.lock().await;
                if voice.is_some() {
                    println!("   • Voice: ✅ Enabled (STT + TTS ready)");
                } else {
                    println!("   • Voice: ⚠️ Enabled but failed to initialize");
                }
            } else {
                println!("   • Voice: ❌ Disabled (use --voice to enable)");
            }
        }
        #[cfg(not(feature = "voice-tts"))]
        {
            println!("   • Voice: ❌ Not compiled (build with --features voice-tts)");
        }
    }

    // Start background consciousness loop
    let loop_state = Arc::clone(&state);
    tokio::spawn(async move {
        consciousness_loop(loop_state, args.loop_interval, args.sleep_interval).await;
    });

    // Start listening
    if let Some(socket_path) = args.socket {
        // Remove existing socket file
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }

        println!("\n🔌 Listening on Unix socket: {:?}", socket_path);
        println!(
            "   Example: echo '{{\"type\":\"ping\"}}' | nc -U {:?}\n",
            socket_path
        );

        let listener = UnixListener::bind(&socket_path)?;

        loop {
            let (stream, _addr) = listener.accept().await?;
            let state = Arc::clone(&state);
            let security = security.clone();

            tokio::spawn(async move {
                match handle_connection(stream, state, security).await {
                    Ok(shutdown) => {
                        if shutdown {
                            info!("Shutdown requested");
                            std::process::exit(0);
                        }
                    }
                    Err(e) => {
                        error!("Connection error: {}", e);
                    }
                }
            });
        }
    } else if let Some(tcp_addr) = args.tcp {
        println!("\n🔌 Listening on TCP: {}", tcp_addr);
        println!(
            "   Example: echo '{{\"type\":\"ping\"}}' | nc {}\n",
            tcp_addr
        );

        let listener = TcpListener::bind(&tcp_addr).await?;

        loop {
            let (stream, addr) = listener.accept().await?;
            info!("New connection from {}", addr);
            let state = Arc::clone(&state);
            let security = security.clone();

            tokio::spawn(async move {
                match handle_connection(stream, state, security).await {
                    Ok(shutdown) => {
                        if shutdown {
                            info!("Shutdown requested");
                            std::process::exit(0);
                        }
                    }
                    Err(e) => {
                        error!("Connection error: {}", e);
                    }
                }
            });
        }
    }

    Ok(())
}
