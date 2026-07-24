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

    /// HTTP gateway address (host:port). Serves the same JSON wire
    /// protocol over `POST /v1/service`, plus `GET /health` and
    /// `GET /metrics` (Prometheus; needs the `api_module` feature).
    /// Runs alongside --socket/--tcp and is subject to the same
    /// refusal-to-bind-non-loopback-without-auth policy as --tcp.
    /// Phase 1 of SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md.
    #[arg(long)]
    http: Option<String>,

    /// Background consciousness loop interval (ms)
    #[arg(long, default_value = "5000")]
    loop_interval: u64,

    /// Auto-sleep interval (seconds, 0 to disable)
    #[arg(long, default_value = "3600")]
    sleep_interval: u64,

    /// State file for persistence
    #[arg(long)]
    state_file: Option<PathBuf>,

    /// SQLite consciousness database path. Enables the persist→recall→
    /// re-perceive experience loop (memory recall, episodic persistence,
    /// Polymath consolidation). Falls back to SYMTHAEA_DATABASE_PATH env
    /// var; without either, the daemon runs amnesiac (in-memory only).
    #[arg(long)]
    database: Option<PathBuf>,

    /// Enable the experience bridge to the autonomous cognitive loop (AGW
    /// Phase 3): drives one CognitiveLoopService::cycle() per process()
    /// call so the loop's knowledge graph and episodic memory accumulate
    /// conversational experience, and reads it back into ethics evaluation
    /// the same turn. When --database is also set, the loop's knowledge
    /// store shares that same SQLite file (survives restarts).
    #[arg(long)]
    experience_bridge: bool,

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

    /// STT worker program (JSONL provider contract, e.g. the Whisper worker:
    /// communication/worker/run_whisper_nixos.sh). Enables the voice_transcribe
    /// request — the semantic-ear lane: audio → Whisper → Symthaea::process().
    #[cfg(feature = "voice-tts")]
    #[arg(long)]
    stt_worker: Option<std::path::PathBuf>,

    /// Extra arguments passed to the STT worker program (space-separated).
    #[cfg(feature = "voice-tts")]
    #[arg(long, default_value = "")]
    stt_worker_args: String,
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

    /// Transcribe a WAV file via the configured STT worker (Whisper lane),
    /// feed the transcript to cognition, and speak the response if voice
    /// output is enabled.
    #[serde(rename = "voice_transcribe")]
    VoiceTranscribe {
        /// Path to a mono WAV file (16 kHz recommended for Whisper).
        audio_path: String,
        /// Expected language hint (e.g. "en"); optional.
        #[serde(default)]
        language: Option<String>,
    },

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

/// Load a WAV file as mono f32 samples (+ sample rate). Multi-channel input
/// is averaged down to mono; integer formats are normalized to [-1, 1].
#[cfg(feature = "voice-tts")]
fn load_wav_mono_f32(path: &str) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };

    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, spec.sample_rate))
}

/// Linear resampler — WAV files handed to `voice_transcribe` may arrive at
/// any native rate (e.g. Kokoro output at 24kHz), but Whisper's feature
/// extractor hard-requires exactly 16kHz input and errors otherwise. Found
/// live 2026-07-17 (LF3 full-duplex verification): the round-trip WER
/// harness already resamples before transcribing, but this service path did
/// not, so `voice_transcribe` failed on any non-16kHz WAV.
#[cfg(feature = "voice-tts")]
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..output_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            match (input.get(idx), input.get(idx + 1)) {
                (Some(&a), Some(&b)) => a * (1.0 - frac) + b * frac,
                (Some(&a), None) => a,
                _ => 0.0,
            }
        })
        .collect()
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
        Request::VoiceTranscribe { .. } => "voice_transcribe",
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
/// Wire form of [`symthaea::CreativeArtifact`]: WAV bytes are base64'd for
/// JSON transport; SVG travels as text.
#[derive(Debug, Serialize)]
#[serde(tag = "kind")]
enum CreativeArtifactWire {
    #[serde(rename = "svg")]
    Svg {
        svg: String,
        aesthetic_composite: f32,
    },
    #[serde(rename = "music_wav")]
    MusicWav {
        /// base64 of complete RIFF/WAVE bytes.
        wav_b64: String,
        duration_secs: f32,
        aesthetic_composite: f32,
    },
}

impl From<&symthaea::CreativeArtifact> for CreativeArtifactWire {
    fn from(artifact: &symthaea::CreativeArtifact) -> Self {
        match artifact {
            symthaea::CreativeArtifact::Svg {
                svg,
                aesthetic_composite,
            } => Self::Svg {
                svg: svg.clone(),
                aesthetic_composite: *aesthetic_composite,
            },
            symthaea::CreativeArtifact::MusicWav {
                wav_bytes,
                duration_secs,
                aesthetic_composite,
            } => {
                #[cfg(feature = "api_module")]
                let wav_b64 = {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD.encode(wav_bytes)
                };
                // Without api_module there is no base64 dep; the service
                // binary always builds with it in practice (--http), but
                // keep the non-api_module build honest rather than broken.
                #[cfg(not(feature = "api_module"))]
                let wav_b64 = {
                    let _ = wav_bytes;
                    String::new()
                };
                Self::MusicWav {
                    wav_b64,
                    duration_secs: *duration_secs,
                    aesthetic_composite: *aesthetic_composite,
                }
            }
        }
    }
}

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
        /// Creative artifact generated when the input expressed art intent
        /// (facade Phase 8.5, feature `creative`). Additive and absent when
        /// no artifact was produced — this field previously had zero
        /// consumers anywhere (VISION_PROJECTION_REVIEW_2026-07-15.md P1.3).
        #[serde(skip_serializing_if = "Option::is_none")]
        creative_artifact: Option<CreativeArtifactWire>,
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
    /// STT worker (Whisper JSONL provider) — the semantic-ear lane.
    /// None unless --stt-worker was passed.
    #[cfg(feature = "voice-tts")]
    stt_provider: Mutex<Option<symthaea_communication::human::LocalJsonlProvider>>,
    /// Broadcasts the experience bridge's cycle telemetry after each query
    /// that actually drove a cycle (SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md
    /// Phase 2). Always constructed — a send with zero subscribers is a
    /// cheap no-op, so there is no need to gate this on whether --http
    /// was actually passed.
    #[cfg(feature = "api_module")]
    telemetry_tx: tokio::sync::broadcast::Sender<LiveTelemetry>,
}

/// Payload for `/v1/ws/live`: cycle metadata plus the projection exits that
/// previously dead-ended inside `CycleResult` with zero consumers — the live
/// cognitive self-portrait (`canvas_svg`) and the imagination decode
/// (`mental_movie`). See VISION_PROJECTION_REVIEW_2026-07-15.md P1.2.
#[cfg(feature = "api_module")]
#[derive(Clone, serde::Serialize)]
struct LiveTelemetry {
    /// Flattened so the wire JSON keeps `CycleMetadata`'s flat key layout —
    /// pre-existing consumers (symthaea-ui `Vitals::from_json`) parse
    /// top-level keys and must keep working; the projection fields below are
    /// additive keys beside them.
    #[serde(flatten)]
    metadata: symthaea::cognitive_loop::CycleMetadata,
    /// Live cognitive self-portrait (animated SVG), when the canvas ticked
    /// this turn. Feature `canvas`; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    canvas_svg: Option<String>,
    /// Geodesic mental-simulation frames, when imagination fired this turn.
    /// Feature `vision-manifold`; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    mental_movie: Option<MentalMovieWire>,
}

/// Wire form of `MentalMovie`: raw frames base64-encoded, HDC trajectory
/// dropped (16,384-D vectors are diagnostic data, not display data).
#[cfg(feature = "api_module")]
#[derive(Clone, serde::Serialize)]
struct MentalMovieWire {
    width: u32,
    height: u32,
    channels: usize,
    semantic_coherence: f32,
    /// Each entry is one frame: base64 of `width*height*channels` raw bytes.
    frames_b64: Vec<String>,
}

#[cfg(feature = "api_module")]
impl LiveTelemetry {
    fn from_cycle(cycle: &symthaea::cognitive_loop::CycleResult) -> Self {
        #[cfg(feature = "vision-manifold")]
        let mental_movie = cycle.mental_movie.as_ref().map(|m| {
            use base64::Engine as _;
            let engine = base64::engine::general_purpose::STANDARD;
            MentalMovieWire {
                width: m.width,
                height: m.height,
                channels: m.channels,
                semantic_coherence: m.semantic_coherence,
                frames_b64: m.frames.iter().map(|f| engine.encode(f)).collect(),
            }
        });
        #[cfg(not(feature = "vision-manifold"))]
        let mental_movie = None;

        #[cfg(feature = "canvas")]
        let canvas_svg = cycle.canvas_svg.clone();
        #[cfg(not(feature = "canvas"))]
        let canvas_svg = None;

        Self {
            metadata: cycle.metadata.clone(),
            canvas_svg,
            mental_movie,
        }
    }
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
        database_path: Option<PathBuf>,
        experience_bridge: bool,
        #[cfg(feature = "voice-tts")] voice_enabled: bool,
        #[cfg(feature = "voice-tts")] voice_id: u8,
        #[cfg(feature = "voice-tts")] stt_worker: Option<PathBuf>,
        #[cfg(feature = "voice-tts")] stt_worker_args: String,
    ) -> Result<Self> {
        // Try to resume from state file if it exists
        let mut symthaea = if let Some(ref path) = state_file {
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

        // Attach the consciousness database. This is what activates the
        // persist→recall→re-perceive experience loop (memory recall via
        // search_similar, episodic/WM persistence, Polymath consolidation)
        // that is otherwise silently dark. Deliberately a HARD error on
        // failure: a daemon asked to persist must not fall back to running
        // amnesiac — that silent degradation is exactly how this gap went
        // unnoticed (AGW plan Phase 1, 2026-07-09).
        if let Some(ref db_path) = database_path {
            let config = symthaea::databases::DatabaseConfig {
                backend: symthaea::databases::DatabaseBackend::Sqlite,
                path: Some(db_path.to_string_lossy().into_owned()),
            };
            symthaea.attach_database(config).await.with_context(|| {
                format!("Failed to attach consciousness database at {:?}", db_path)
            })?;
            info!("Consciousness database attached: {:?}", db_path);
        } else {
            warn!(
                "No --database / SYMTHAEA_DATABASE_PATH configured — running amnesiac (no memory persistence)"
            );
        }

        // AGW Phase 3: experience bridge to the autonomous cognitive loop.
        // Reuses the same SQLite file as --database when set (Option A1) so
        // the loop's knowledge graph survives restarts in the same store.
        if experience_bridge {
            let knowledge_db_path = database_path
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned());
            match symthaea.enable_experience_bridge(knowledge_db_path) {
                Ok(()) => info!("Experience bridge to the autonomous cognitive loop enabled"),
                Err(e) => error!("Failed to enable experience bridge: {}", e),
            }
        }

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

        // Spawn the STT worker (Whisper lane) if configured. The provider
        // keeps the worker process alive across requests (model loads once).
        #[cfg(feature = "voice-tts")]
        let stt_provider = if let Some(ref worker) = stt_worker {
            use symthaea_communication::human::{LocalJsonlProvider, WorkerPolicy};
            let args: Vec<String> = stt_worker_args
                .split_whitespace()
                .map(str::to_string)
                .collect();
            match WorkerPolicy::allow_one(worker)
                .and_then(|policy| LocalJsonlProvider::spawn("service-stt", worker, &args, policy))
            {
                Ok(provider) => {
                    info!("STT worker ready: {}", worker.display());
                    Some(provider)
                }
                Err(e) => {
                    warn!("Failed to start STT worker {}: {}", worker.display(), e);
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
            #[cfg(feature = "voice-tts")]
            stt_provider: Mutex::new(stt_provider),
            // Capacity 16 — telemetry is a "latest state" feed; a slow
            // subscriber should drop stale cycles, not backpressure the
            // query path.
            #[cfg(feature = "api_module")]
            telemetry_tx: tokio::sync::broadcast::channel(16).0,
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

                        // Phase 2 (SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md):
                        // if the experience bridge cycled this turn, publish
                        // its telemetry. `send` errors only when there are
                        // zero subscribers — expected and harmless.
                        #[cfg(feature = "api_module")]
                        if let Some(cycle) = symthaea.last_bridge_cycle() {
                            let _ = self.telemetry_tx.send(LiveTelemetry::from_cycle(cycle));
                        }

                        Response::QueryResponse {
                            content: response.content,
                            confidence: response.confidence,
                            safe: response.safe,
                            phi: intro.consciousness_level,
                            phi_dyad: partnership.phi_dyad,
                            steps_to_emergence: response.steps_to_emergence,
                            processing_time_ms: start.elapsed().as_millis() as u64,
                            creative_artifact: response
                                .creative_artifact
                                .as_ref()
                                .map(CreativeArtifactWire::from),
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
                        if let Some(ref mut voice) = *voice
                            && let Err(e) = voice.speak(&assistant_said)
                        {
                            warn!("TTS error (continuing): {}", e);
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

            Request::VoiceTranscribe {
                audio_path,
                language,
            } => {
                #[cfg(feature = "voice-tts")]
                {
                    // 1. Load the WAV file (mono-ize, convert to f32), then
                    //    resample to 16kHz — Whisper hard-requires it and
                    //    the source WAV may be at any native rate.
                    const WHISPER_SAMPLE_RATE: u32 = 16_000;
                    let (samples, sample_rate) = match load_wav_mono_f32(&audio_path) {
                        Ok((samples, rate)) => (
                            resample_linear(&samples, rate, WHISPER_SAMPLE_RATE),
                            WHISPER_SAMPLE_RATE,
                        ),
                        Err(e) => {
                            return Response::Error {
                                message: format!("Failed to load {audio_path}: {e}"),
                            };
                        }
                    };

                    // 2. Transcribe through the STT JSONL worker (Whisper lane).
                    let user_said = {
                        let mut provider = self.stt_provider.lock().await;
                        let Some(ref mut provider) = *provider else {
                            return Response::Error {
                                message: "No STT worker configured (start the service with \
                                          --stt-worker <path>)"
                                    .into(),
                            };
                        };
                        match symthaea::voice::transcribe_samples(
                            provider,
                            samples,
                            sample_rate,
                            language.as_deref(),
                        ) {
                            Ok(preserved) => preserved.original,
                            Err(e) => {
                                return Response::Error {
                                    message: format!("Transcription failed: {e}"),
                                };
                            }
                        }
                    };

                    if user_said.trim().is_empty() {
                        return Response::Error {
                            message: "Transcription produced no text".into(),
                        };
                    }

                    // 3. Feed the transcript to cognition (the semantic ear
                    //    finally reaching Symthaea::process()).
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
                                    message: format!("Processing error: {e}"),
                                };
                            }
                        }
                    };

                    // 4. Speak the reply if voice output is enabled.
                    {
                        let mut voice = self.voice.lock().await;
                        if let Some(ref mut voice) = *voice
                            && let Err(e) = voice.speak(&assistant_said)
                        {
                            warn!("TTS error (continuing): {}", e);
                        }
                    }

                    self.record_audit("voice_transcribe", &audio_path, &user_said);

                    Response::VoiceTurnResponse {
                        user_said,
                        assistant_said,
                        phi,
                        processing_time_ms: start.elapsed().as_millis() as u64,
                    }
                }
                #[cfg(not(feature = "voice-tts"))]
                {
                    let _ = (audio_path, language);
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
                    // stt_ready reflects the actual transcription lane (the
                    // Whisper worker), not TTS session presence.
                    let stt_ready = self.stt_provider.lock().await.is_some();
                    Response::VoiceStatusResponse {
                        enabled: self.voice_enabled,
                        stt_ready,
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

    if line.len() > MAX_REQUEST_LINE_BYTES {
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

    if let Some(version) = wire_request.protocol_version
        && version != SERVICE_PROTOCOL_VERSION
    {
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

    if request_requires_auth(&wire_request.request)
        && let Some(expected_token) = security.bearer_token.as_deref()
    {
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

/// Shared context for the HTTP gateway handlers.
#[derive(Clone)]
struct HttpGatewayCtx {
    state: Arc<ServiceState>,
    security: ServiceSecurity,
}

/// Fold a standard `Authorization` header into the wire envelope's
/// top-level `authorization` field. The body's own field wins when both
/// are present. Non-JSON and non-object bodies pass through untouched so
/// `process_request_line` reports the same invalid-JSON error the socket
/// path would.
fn fold_authorization_header(body: &str, headers: &axum::http::HeaderMap) -> String {
    let Some(header_value) = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return body.to_string();
    };
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(mut value) => {
            if let Some(obj) = value.as_object_mut() {
                obj.entry("authorization")
                    .or_insert_with(|| serde_json::Value::String(header_value.to_string()));
                return value.to_string();
            }
            body.to_string()
        }
        Err(_) => body.to_string(),
    }
}

/// Map a wire-protocol response onto an HTTP status code. The envelope
/// keeps its own error semantics (`{"type":"error",...}`); this mapping
/// only exists so plain HTTP clients can branch without parsing.
fn http_status_for_response(response: &Response) -> axum::http::StatusCode {
    use axum::http::StatusCode;
    match response {
        Response::Error { message } if message == &service_auth_error_message() => {
            StatusCode::UNAUTHORIZED
        }
        Response::Error { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::OK,
    }
}

async fn http_health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "status": "ok",
        "service": "symthaea-service",
        "protocol_version": SERVICE_PROTOCOL_VERSION,
    }))
}

async fn http_metrics() -> axum::response::Response {
    use axum::response::IntoResponse;
    #[cfg(feature = "api_module")]
    {
        let registry = symthaea::api::metrics::global();
        registry.increment("api_requests_total");
        (
            axum::http::StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            registry.to_prometheus_text(),
        )
            .into_response()
    }
    #[cfg(not(feature = "api_module"))]
    {
        (
            axum::http::StatusCode::NOT_IMPLEMENTED,
            "metrics require a build with the api_module feature",
        )
            .into_response()
    }
}

async fn http_service_endpoint(
    axum::extract::State(ctx): axum::extract::State<HttpGatewayCtx>,
    headers: axum::http::HeaderMap,
    body: String,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let line = fold_authorization_header(&body, &headers);
    match process_request_line(&line, ctx.state.clone(), ctx.security.clone()).await {
        Ok(Some(outcome)) => {
            if outcome.shutdown {
                // Mirror the socket path: respond, then exit. The short
                // delay lets the response flush before the process dies.
                tokio::spawn(async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    info!("Shutdown requested via HTTP gateway");
                    std::process::exit(0);
                });
            }
            let status = http_status_for_response(&outcome.response);
            match serde_json::to_value(&outcome.response) {
                Ok(json) => (status, axum::Json(json)).into_response(),
                Err(e) => (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to serialize response: {e}"),
                )
                    .into_response(),
            }
        }
        Ok(None) => (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "type": "error",
                "message": "Empty request body",
            })),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "type": "error",
                "message": format!("Internal error: {e}"),
            })),
        )
            .into_response(),
    }
}

/// `GET /v1/ws/live` (Phase 2 of SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md).
/// Streams the experience bridge's cycle telemetry as it happens — one
/// message per query that actually drove a cycle, not a synthetic clock.
/// Wire type is `LiveTelemetry`: the raw `CycleMetadata` plus the
/// projection exits (`canvas_svg` self-portrait, base64 `mental_movie`
/// frames) that previously had zero consumers. It is deliberately not the
/// demo binary's curated `DemoCycleData`: producing that richer projection
/// needs several live `CognitiveLoopService` accessors (swarm/mesh/bath
/// tracker/ethics-topology state) that `demo_runner.rs` currently builds
/// inline rather than through a reusable function — factoring that out is
/// deliberately left as a follow-up rather than done as a byproduct of
/// this gateway, since it touches a large, actively-developed mapping.
#[cfg(feature = "api_module")]
async fn http_ws_live(
    axum::extract::State(ctx): axum::extract::State<HttpGatewayCtx>,
    headers: axum::http::HeaderMap,
    ws: axum::extract::ws::WebSocketUpgrade,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    // The WS handshake is a plain HTTP GET before the protocol switches,
    // so the same bearer check as /v1/service applies here — non-browser
    // WS clients can set the header on the handshake request.
    if let Some(expected) = ctx.security.bearer_token.as_deref() {
        let provided = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(parse_bearer_token);
        if provided.as_deref() != Some(expected) {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                service_auth_error_message(),
            )
                .into_response();
        }
    }

    let rx = ctx.state.telemetry_tx.subscribe();
    ws.on_upgrade(move |socket| telemetry_ws_loop(socket, rx))
}

#[cfg(feature = "api_module")]
async fn telemetry_ws_loop(
    mut socket: axum::extract::ws::WebSocket,
    mut rx: tokio::sync::broadcast::Receiver<LiveTelemetry>,
) {
    use axum::extract::ws::Message;
    loop {
        match rx.recv().await {
            Ok(telemetry) => {
                let Ok(payload) = serde_json::to_string(&telemetry) else {
                    continue;
                };
                if socket.send(Message::Text(payload.into())).await.is_err() {
                    break; // client disconnected
                }
            }
            // A lagging subscriber missed some cycles — telemetry is a
            // "latest state" feed, so skip forward rather than disconnect.
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
}

/// Build the HTTP gateway router (Phase 1 of
/// SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md). One route carries the whole
/// existing wire protocol, so auth, protocol versioning, size limits, and
/// audit logging behave identically to the Unix-socket path — this is a
/// transport, not a second protocol.
fn build_http_router(state: Arc<ServiceState>, security: ServiceSecurity) -> axum::Router {
    use axum::routing::{get, post};
    let ctx = HttpGatewayCtx { state, security };
    let router = axum::Router::new()
        .route("/v1/service", post(http_service_endpoint))
        .route("/health", get(http_health))
        .route("/v1/health", get(http_health))
        .route("/metrics", get(http_metrics));
    #[cfg(feature = "api_module")]
    let router = router.route("/v1/ws/live", get(http_ws_live));
    router.with_state(ctx)
}

async fn serve_http(
    addr: String,
    state: Arc<ServiceState>,
    security: ServiceSecurity,
) -> Result<()> {
    let app = build_http_router(state, security);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("Failed to bind HTTP gateway to {addr}"))?;
    axum::serve(listener, app)
        .await
        .context("HTTP gateway server error")?;
    Ok(())
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
                None,
                false,
                #[cfg(feature = "voice-tts")]
                false,
                #[cfg(feature = "voice-tts")]
                0,
                #[cfg(feature = "voice-tts")]
                None,
                #[cfg(feature = "voice-tts")]
                String::new(),
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

    // ── HTTP gateway (Phase 1, SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md) ──

    #[test]
    fn authorization_header_folds_into_envelope() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer header-token".parse().expect("header value"),
        );

        let folded = fold_authorization_header(r#"{"type":"status"}"#, &headers);
        let value: serde_json::Value = serde_json::from_str(&folded).expect("folded JSON");
        assert_eq!(value["authorization"], "Bearer header-token");

        // The body's own authorization field wins over the header.
        let folded = fold_authorization_header(
            r#"{"type":"status","authorization":"Bearer body-token"}"#,
            &headers,
        );
        let value: serde_json::Value = serde_json::from_str(&folded).expect("folded JSON");
        assert_eq!(value["authorization"], "Bearer body-token");

        // Invalid JSON passes through untouched (process_request_line
        // then reports the same error the socket path would).
        assert_eq!(fold_authorization_header("not json", &headers), "not json");
    }

    #[tokio::test]
    async fn http_gateway_health_is_public() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use tower::util::ServiceExt;

        let app = build_http_router(test_state(true).await, test_security(Some("tok")));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response");
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn http_gateway_serves_wire_protocol() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use tower::util::ServiceExt;

        let app = build_http_router(test_state(false).await, test_security(None));
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method(axum::http::Method::POST)
                    .uri("/v1/service")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"ping"}"#))
                    .expect("ping request"),
            )
            .await
            .expect("ping response");
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("pong JSON");
        assert_eq!(json["type"], "pong");
    }

    #[tokio::test]
    async fn http_gateway_enforces_bearer_auth() {
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use tower::util::ServiceExt;

        let state = test_state(true).await;
        let security = test_security(Some("secret"));

        // Auth-required request without a token → 401.
        let response = build_http_router(state.clone(), security.clone())
            .oneshot(
                HttpRequest::builder()
                    .method(axum::http::Method::POST)
                    .uri("/v1/service")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"status"}"#))
                    .expect("status request"),
            )
            .await
            .expect("status response");
        assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED);

        // Same request with the Authorization header folded in → 200.
        let response = build_http_router(state, security)
            .oneshot(
                HttpRequest::builder()
                    .method(axum::http::Method::POST)
                    .uri("/v1/service")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer secret")
                    .body(Body::from(r#"{"type":"status"}"#))
                    .expect("status request"),
            )
            .await
            .expect("status response");
        assert_eq!(response.status(), axum::http::StatusCode::OK);
    }

    #[cfg(feature = "api_module")]
    #[tokio::test]
    async fn telemetry_broadcasts_after_a_bridge_driven_query() {
        let state = test_state(false).await;

        // No experience bridge yet — a query must not publish telemetry.
        let mut rx_before = state.telemetry_tx.subscribe();
        let _ = process_json(
            json!({"type": "query", "content": "no bridge yet"}),
            state.clone(),
            test_security(None),
        )
        .await;
        assert!(
            rx_before.try_recv().is_err(),
            "no telemetry should publish before the bridge is enabled"
        );

        // Enable the bridge, then a query must publish exactly one cycle.
        {
            let mut symthaea = state.symthaea.lock().await;
            symthaea
                .enable_experience_bridge(None)
                .expect("in-memory experience bridge must construct cleanly");
        }
        let mut rx = state.telemetry_tx.subscribe();
        let _ = process_json(
            json!({"type": "query", "content": "bridge is live now"}),
            state.clone(),
            test_security(None),
        )
        .await;

        let metadata = tokio::time::timeout(Duration::from_secs(10), rx.recv())
            .await
            .expect("telemetry received within timeout")
            .expect("telemetry channel open");
        // CycleMetadata isn't PartialEq; a successful, type-checked receive
        // off the broadcast channel is the assertion — confirms the wire
        // path from bridge cycle to subscriber, not any specific value.
        let _ = serde_json::to_value(&metadata).expect("CycleMetadata serializes");
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
    if args.socket.is_none() && args.tcp.is_none() && args.http.is_none() {
        anyhow::bail!("Must specify at least one of --socket, --tcp, or --http");
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

    // Same refusal policy for every TCP-based transport (--tcp and --http).
    for addr in [args.tcp.as_ref(), args.http.as_ref()]
        .into_iter()
        .flatten()
    {
        if !addr_is_loopback(addr) && service_bearer_token.is_none() && !insecure_allow_unauth {
            eprintln!("Refusing to bind Symthaea service to non-loopback address without auth.");
            eprintln!("  addr: {}", addr);
            eprintln!();
            eprintln!("Set one of:");
            eprintln!("  - SYMTHAEA_SERVICE_BEARER_TOKEN=...   (recommended)");
            eprintln!("  - SYMTHAEA_SERVICE_INSECURE_ALLOW_UNAUTH=1   (NOT recommended)");
            std::process::exit(2);
        }

        if !addr_is_loopback(addr) && service_bearer_token.is_none() && insecure_allow_unauth {
            eprintln!("WARNING: Symthaea service is binding publicly without auth (insecure).");
            eprintln!("  addr: {}", addr);
        }
    }

    println!("\n🌟 Symthaea Service Starting...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Initialize state
    info!("Initializing consciousness...");
    // --database flag wins; SYMTHAEA_DATABASE_PATH env var is the fallback
    // so the systemd unit can enable persistence via Environment= alone.
    let database_path = args.database.clone().or_else(|| {
        std::env::var("SYMTHAEA_DATABASE_PATH")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    });
    let experience_bridge = args.experience_bridge
        || std::env::var("SYMTHAEA_EXPERIENCE_BRIDGE")
            .map(|v| v == "1")
            .unwrap_or(false);

    let state = Arc::new(
        ServiceState::new(
            service_bearer_token.is_some(),
            service_audit_log_path,
            args.state_file.clone(),
            database_path,
            experience_bridge,
            #[cfg(feature = "voice-tts")]
            args.voice,
            #[cfg(feature = "voice-tts")]
            args.voice_id,
            #[cfg(feature = "voice-tts")]
            args.stt_worker.clone(),
            #[cfg(feature = "voice-tts")]
            args.stt_worker_args.clone(),
        )
        .await
        .context("Failed to initialize service state")?,
    );
    let security = ServiceSecurity {
        bearer_token: service_bearer_token,
    };

    // systemd stops this daemon with SIGTERM. Without a handler the process
    // died without saving relational/partnership state, silently defeating
    // --state-file on every `systemctl restart` — state was only saved on an
    // explicit wire `shutdown` request, which nothing sends in production
    // (found during AGW Phase 1 verification, 2026-07-09).
    #[cfg(unix)]
    {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! {
                _ = sigterm.recv() => {},
                _ = tokio::signal::ctrl_c() => {},
            }
            if let Some(ref path) = state.state_file {
                let path_str = path.to_string_lossy();
                let mut symthaea = state.symthaea.lock().await;
                match symthaea.pause(&path_str) {
                    Ok(()) => info!("Shutdown signal: state saved to {:?}", path),
                    Err(e) => error!("Shutdown signal: failed to save state: {}", e),
                }
            }
            std::process::exit(0);
        });
    }

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

    // HTTP gateway — runs alongside the socket/TCP listeners.
    let http_handle = if let Some(http_addr) = args.http.clone() {
        let state = Arc::clone(&state);
        let security = security.clone();
        println!("\n🌐 HTTP gateway listening on http://{}", http_addr);
        println!("   POST /v1/service  |  GET /health  |  GET /metrics");
        println!(
            "   Example: curl -s http://{}/v1/service -d '{{\"type\":\"ping\"}}'\n",
            http_addr
        );
        Some(tokio::spawn(async move {
            if let Err(e) = serve_http(http_addr, state, security).await {
                error!("HTTP gateway failed: {}", e);
            }
        }))
    } else {
        None
    };

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
    } else if let Some(handle) = http_handle {
        // HTTP-only mode: the gateway task is what keeps the daemon alive.
        handle.await?;
    }

    Ok(())
}
