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
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::RwLock;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use symthaea::hdc::{HDC_DIMENSION, LTC_NEURONS};
use symthaea::Symthaea;

// Voice support (feature-gated)
#[cfg(feature = "voice")]
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
    #[cfg(feature = "voice")]
    #[arg(long)]
    voice: bool,

    /// Voice input device (default: system default)
    #[cfg(feature = "voice")]
    #[arg(long)]
    voice_input: Option<String>,

    /// Voice ID for TTS (0-9)
    #[cfg(feature = "voice")]
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
#[allow(dead_code)]  // Fields used via serde deserialization
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

// Default value helpers for serde
fn default_phi_threshold() -> f32 { 0.5 }
fn default_true() -> bool { true }
fn default_metrics_interval() -> u64 { 1000 }
fn default_search_limit() -> usize { 10 }

/// Response to client
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
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

    /// Error response
    #[serde(rename = "error")]
    Error { message: String },

    /// Speech synthesized (TTS complete)
    #[serde(rename = "spoken")]
    Spoken { text: String, duration_ms: u64 },

    /// Speech transcribed (STT complete)
    #[serde(rename = "transcribed")]
    Transcribed { text: String, confidence: f32 },

    /// Voice conversation turn complete
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

    // ========================================================================
    // GUI BRIDGE RESPONSES (Phase 4 Protocol Extensions)
    // ========================================================================

    /// GUI synchronization response
    #[serde(rename = "gui_sync")]
    GuiSync {
        /// Widget state updates
        widget_updates: Vec<WidgetUpdate>,
        /// Generated Nix diff (if applicable)
        nix_diff: Option<String>,
        /// Validation errors
        validation_errors: Vec<ValidationError>,
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

// ============================================================================
// SUPPORTING TYPES FOR GUI BRIDGE
// ============================================================================

/// Widget state update for GUI synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetUpdate {
    /// Widget identifier
    pub widget_id: String,
    /// New value
    pub value: serde_json::Value,
    /// Source of the update
    pub source: UpdateSource,
}

/// Source of a widget update
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdateSource {
    NixConfig,
    UserInput,
    Default,
}

/// Validation error for GUI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Widget or path that has the error
    pub target: String,
    /// Error message
    pub message: String,
    /// Severity
    pub severity: ErrorSeverity,
    /// Suggested fix
    pub suggested_fix: Option<String>,
}

/// Error severity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorSeverity {
    Error,
    Warning,
    Info,
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
    symthaea: Symthaea,
    start_time: Instant,
    requests_processed: u64,
    sleep_cycles: u32,
    state_file: Option<PathBuf>,
    #[cfg(feature = "voice")]
    voice: Option<VoiceConversation>,
    #[cfg(feature = "voice")]
    voice_enabled: bool,
}

impl ServiceState {
    async fn new(
        state_file: Option<PathBuf>,
        #[cfg(feature = "voice")] voice_enabled: bool,
        #[cfg(feature = "voice")] voice_id: u8,
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
        #[cfg(feature = "voice")]
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
            symthaea,
            start_time: Instant::now(),
            requests_processed: 0,
            sleep_cycles: 0,
            state_file,
            #[cfg(feature = "voice")]
            voice,
            #[cfg(feature = "voice")]
            voice_enabled,
        })
    }

    async fn handle_request(&mut self, request: Request) -> Response {
        self.requests_processed += 1;

        match request {
            Request::Query { content, context: _ } => {
                let start = Instant::now();
                match self.symthaea.process(&content).await {
                    Ok(response) => {
                        let intro = self.symthaea.introspect();
                        let partnership = self.symthaea.partnership_state();
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
                let intro = self.symthaea.introspect();
                Response::Status {
                    uptime_seconds: self.start_time.elapsed().as_secs(),
                    requests_processed: self.requests_processed,
                    consciousness_level: intro.consciousness_level,
                    memory_count: intro.memory_stats.short_term_count
                        + intro.memory_stats.long_term_count,
                    sleep_cycles: self.sleep_cycles,
                }
            }

            Request::Introspect => {
                let intro = self.symthaea.introspect();
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
                    phenomenal_state: if is_conscious { "Aware".to_string() } else { "Dormant".to_string() },
                    cycles_since_awakening: 0, // Not tracked yet
                    self_model_accuracy: intro.complexity as f64 / 10.0,
                }
            }

            Request::Sleep => match self.symthaea.sleep().await {
                Ok(report) => {
                    self.sleep_cycles += 1;
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
            },

            Request::Save { path } => {
                let save_path = path
                    .map(PathBuf::from)
                    .or_else(|| self.state_file.clone())
                    .unwrap_or_else(|| PathBuf::from("symthaea-state.bin"));

                let path_str = save_path.to_string_lossy();
                match self.symthaea.pause(&path_str) {
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
                    let _ = self.symthaea.pause(&path_str);
                    info!("State saved to {:?}", path);
                }
                Response::ShutdownAck
            }

            Request::Ping => Response::Pong {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },

            // Voice requests
            Request::Speak { text } => {
                #[cfg(feature = "voice")]
                {
                    if let Some(ref mut voice) = self.voice {
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
                #[cfg(not(feature = "voice"))]
                {
                    let _ = text;
                    Response::Error {
                        message: "Voice feature not compiled".into(),
                    }
                }
            }

            Request::Listen => {
                #[cfg(feature = "voice")]
                {
                    if let Some(ref mut voice) = self.voice {
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
                #[cfg(not(feature = "voice"))]
                {
                    Response::Error {
                        message: "Voice feature not compiled".into(),
                    }
                }
            }

            Request::VoiceTurn => {
                #[cfg(feature = "voice")]
                {
                    if let Some(ref mut voice) = self.voice {
                        let start = Instant::now();

                        // Listen for user speech
                        let user_said = match voice.listen() {
                            Ok(text) => text,
                            Err(e) => {
                                return Response::Error {
                                    message: format!("Listen error: {}", e),
                                };
                            }
                        };

                        // Process through consciousness
                        let (assistant_said, phi) = match self.symthaea.process(&user_said).await {
                            Ok(response) => {
                                let intro = self.symthaea.introspect();
                                (response.content, intro.consciousness_level)
                            }
                            Err(e) => {
                                return Response::Error {
                                    message: format!("Processing error: {}", e),
                                };
                            }
                        };

                        // Speak response
                        if let Err(e) = voice.speak(&assistant_said) {
                            warn!("TTS error (continuing): {}", e);
                        }

                        Response::VoiceTurnResponse {
                            user_said,
                            assistant_said,
                            phi,
                            processing_time_ms: start.elapsed().as_millis() as u64,
                        }
                    } else {
                        Response::Error {
                            message: "Voice not enabled".into(),
                        }
                    }
                }
                #[cfg(not(feature = "voice"))]
                {
                    Response::Error {
                        message: "Voice feature not compiled".into(),
                    }
                }
            }

            Request::VoiceStatus => {
                #[cfg(feature = "voice")]
                {
                    Response::VoiceStatusResponse {
                        enabled: self.voice_enabled,
                        stt_ready: self.voice.is_some(),
                        tts_ready: self.voice.is_some(),
                        voice_id: self.voice.as_ref().map(|_| 0).unwrap_or(0),
                    }
                }
                #[cfg(not(feature = "voice"))]
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
                let state = self.symthaea.partnership_state();
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

            Request::IntelliSense { partial_input, cursor_position: _, context: _ } => {
                let intro = self.symthaea.introspect();

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

            Request::ValidateCommand { command, dry_run: _ } => {
                use symthaea::action::{classify_command_destructiveness, get_rollback_hint, DestructivenessLevel};

                let intro = self.symthaea.introspect();

                // Parse command into program and args
                let parts: Vec<&str> = command.split_whitespace().collect();
                let (program, args) = if parts.is_empty() {
                    ("", vec![])
                } else {
                    (parts[0], parts[1..].iter().map(|s| s.to_string()).collect())
                };

                // Classify destructiveness
                let destructiveness = classify_command_destructiveness(program, &args);
                let rollback_hint = get_rollback_hint(program, &args);

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
                if destructiveness.requires_confirmation() {
                    warnings.push(format!("This command requires confirmation: {}",
                        destructiveness.description()));
                }
                if current_phi < phi_required as f64 {
                    warnings.push(format!(
                        "Current Phi ({:.2}) is below required threshold ({:.2})",
                        current_phi, phi_required
                    ));
                }

                Response::ValidationResult {
                    valid: true,
                    safety_level,
                    destructiveness: format!("{:?}", destructiveness),
                    phi_required,
                    warnings,
                    suggested_alternatives: Vec::new(),
                    rollback_hint,
                }
            }

            Request::ExecuteGated { command, phi_threshold, require_confirmation } => {
                use symthaea::action::{classify_command_destructiveness, get_rollback_hint, DestructivenessLevel};

                let intro = self.symthaea.introspect();
                let current_phi = intro.consciousness_level; // Use consciousness_level as phi

                // Parse command
                let parts: Vec<&str> = command.split_whitespace().collect();
                let (program, args) = if parts.is_empty() {
                    ("", vec![])
                } else {
                    (parts[0], parts[1..].iter().map(|s| s.to_string()).collect())
                };

                let destructiveness = classify_command_destructiveness(program, &args);
                let rollback_hint = get_rollback_hint(program, &args);
                let needs_confirmation = require_confirmation && destructiveness.requires_confirmation();

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

                // Execute through consciousness pipeline
                match self.symthaea.process(&command).await {
                    Ok(response) => Response::ExecutionResult {
                        executed: true,
                        output: Some(response.content),
                        phi_at_execution: current_phi,
                        gate_reason: None,
                        requires_confirmation: false,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    },
                    Err(e) => Response::ExecutionResult {
                        executed: false,
                        output: Some(format!("Error: {}", e)),
                        phi_at_execution: current_phi,
                        gate_reason: Some(e.to_string()),
                        requires_confirmation: false,
                        destructiveness: format!("{:?}", destructiveness),
                        rollback_hint,
                    },
                }
            }

            Request::StreamMetrics { interval_ms: _ } => {
                // For now, return a single metrics snapshot
                // Full streaming would require a different connection model
                let intro = self.symthaea.introspect();

                Response::MetricsUpdate {
                    phi: intro.consciousness_level, // Use consciousness_level as phi approximation
                    coherence: intro.consciousness_level,
                    consciousness_level: intro.consciousness_level,
                    safety_checks: self.requests_processed,
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                }
            }

            // ================================================================
            // GUI BRIDGE HANDLERS (Phase 4 - Stub implementations)
            // ================================================================

            Request::GuiWidgetChange { widget_id, new_value: _, semantic_intent: _ } => {
                // Stub: Full implementation in Phase 4
                Response::GuiSync {
                    widget_updates: vec![],
                    nix_diff: Some(format!("# Widget {} changed", widget_id)),
                    validation_errors: vec![],
                }
            }

            Request::ParseNixConfig { nix_content, source_file: _ } => {
                // Stub: Full implementation in Phase 4
                let line_count = nix_content.lines().count();
                Response::GuiSync {
                    widget_updates: vec![],
                    nix_diff: None,
                    validation_errors: if line_count == 0 {
                        vec![ValidationError {
                            target: "config".to_string(),
                            message: "Empty configuration".to_string(),
                            severity: ErrorSeverity::Warning,
                            suggested_fix: None,
                        }]
                    } else {
                        vec![]
                    },
                }
            }

            Request::SemanticSearch { query, search_type, limit } => {
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
    use symthaea::action::{classify_command_destructiveness, DestructivenessLevel};

    let partial_lower = partial.to_lowercase();
    let mut completions = Vec::new();

    // NixOS command completions
    let nix_commands = [
        ("nix search nixpkgs", "Search for packages in nixpkgs", CompletionKind::Command),
        ("nix-env -i", "Install a package to user profile", CompletionKind::Command),
        ("nix-env -q", "Query installed packages", CompletionKind::Command),
        ("nix flake show", "Show flake outputs", CompletionKind::Command),
        ("nixos-rebuild switch", "Rebuild and switch to new configuration", CompletionKind::Command),
        ("nixos-rebuild test", "Build and test configuration without switching", CompletionKind::Command),
        ("nixos-rebuild dry-run", "Show what would be built", CompletionKind::Command),
        ("nix-collect-garbage -d", "Delete old generations (destructive)", CompletionKind::Command),
        ("systemctl status", "Show service status", CompletionKind::Command),
        ("systemctl restart", "Restart a service", CompletionKind::Command),
        ("journalctl -u", "View service logs", CompletionKind::Command),
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
    completions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
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
    use symthaea::action::{classify_command_destructiveness, DestructivenessLevel};

    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        return None;
    }

    let (prog, args): (&str, Vec<String>) = (
        parts[0],
        parts[1..].iter().map(|s| s.to_string()).collect()
    );

    let destructiveness = classify_command_destructiveness(prog, &args);

    // Generate steps based on command type
    let steps = if command.contains("nixos-rebuild") {
        vec![
            CommandStep { number: 1, description: "Evaluate configuration".to_string(), reversible: true },
            CommandStep { number: 2, description: "Build system derivation".to_string(), reversible: true },
            CommandStep { number: 3, description: "Activate new generation".to_string(), reversible: true },
            CommandStep { number: 4, description: "Update boot loader".to_string(), reversible: true },
        ]
    } else if command.contains("nix-env -i") {
        vec![
            CommandStep { number: 1, description: "Resolve package".to_string(), reversible: true },
            CommandStep { number: 2, description: "Download/build package".to_string(), reversible: true },
            CommandStep { number: 3, description: "Install to profile".to_string(), reversible: true },
        ]
    } else if command.contains("nix-collect-garbage") {
        vec![
            CommandStep { number: 1, description: "Find unused store paths".to_string(), reversible: true },
            CommandStep { number: 2, description: "Delete old generations".to_string(), reversible: false },
            CommandStep { number: 3, description: "Remove store paths".to_string(), reversible: false },
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
fn generate_search_results(query: &str, search_type: SearchType, limit: usize) -> Vec<SearchResult> {
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
                        relevance: if name.starts_with(&query_lower) { 0.9 } else { 0.7 },
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

/// Handle a single connection
async fn handle_connection<S>(mut stream: S, state: Arc<RwLock<ServiceState>>) -> Result<bool>
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
            // Connection closed
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        debug!("Received: {}", line);

        // Parse request
        let request: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let response = Response::Error {
                    message: format!("Invalid JSON: {}", e),
                };
                let json = serde_json::to_string(&response)?;
                writer.write_all(json.as_bytes()).await?;
                writer.write_all(b"\n").await?;
                writer.flush().await?;
                continue;
            }
        };

        // Check for shutdown
        let is_shutdown = matches!(request, Request::Shutdown);

        // Handle request
        let response = {
            let mut state = state.write().await;
            state.handle_request(request).await
        };

        // Send response
        let json = serde_json::to_string(&response)?;
        writer.write_all(json.as_bytes()).await?;
        writer.write_all(b"\n").await?;
        writer.flush().await?;

        if is_shutdown {
            return Ok(true); // Signal shutdown
        }
    }

    Ok(false)
}

/// Background consciousness loop
async fn consciousness_loop(state: Arc<RwLock<ServiceState>>, interval_ms: u64, sleep_interval: u64) {
    let mut ticker = interval(Duration::from_millis(interval_ms));
    let mut sleep_counter = 0u64;

    loop {
        ticker.tick().await;

        // Simple consciousness maintenance
        {
            let state = state.read().await;
            let intro = state.symthaea.introspect();
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
                let mut state = state.write().await;
                if let Ok(report) = state.symthaea.sleep().await {
                    state.sleep_cycles += 1;
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

    println!("\n🌟 Symthaea Service Starting...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Initialize state
    info!("Initializing consciousness...");
    let state = Arc::new(RwLock::new(
        ServiceState::new(
            args.state_file.clone(),
            #[cfg(feature = "voice")]
            args.voice,
            #[cfg(feature = "voice")]
            args.voice_id,
        )
        .await
        .context("Failed to initialize service state")?,
    ));

    {
        let s = state.read().await;
        let intro = s.symthaea.introspect();
        // Derive consciousness metrics
        let phi = intro.consciousness_level as f64;
        let is_conscious = intro.consciousness_level > 0.5;

        println!("✅ Consciousness initialized:");
        println!("   • HDC Dimension: {}", HDC_DIMENSION);
        println!("   • LTC Neurons: {}", LTC_NEURONS);
        println!("   • Consciousness Level: {:.1}%", intro.consciousness_level * 100.0);
        println!("   • Graph Size: {} states", intro.graph_size);
        println!("   • Self-Loops: {}", intro.self_loops);
        println!("   • λ₂ (Spectral Connectivity): {:.3}", phi);
        println!("   • Is Conscious: {}", if is_conscious { "✅ Yes" } else { "🔄 Awakening..." });

        #[cfg(feature = "voice")]
        {
            if s.voice_enabled {
                if s.voice.is_some() {
                    println!("   • Voice: ✅ Enabled (STT + TTS ready)");
                } else {
                    println!("   • Voice: ⚠️ Enabled but failed to initialize");
                }
            } else {
                println!("   • Voice: ❌ Disabled (use --voice to enable)");
            }
        }
        #[cfg(not(feature = "voice"))]
        {
            println!("   • Voice: ❌ Not compiled (build with --features voice)");
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
        println!("   Example: echo '{{\"type\":\"ping\"}}' | nc -U {:?}\n", socket_path);

        let listener = UnixListener::bind(&socket_path)?;

        loop {
            let (stream, _addr) = listener.accept().await?;
            let state = Arc::clone(&state);

            tokio::spawn(async move {
                match handle_connection(stream, state).await {
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
        println!("   Example: echo '{{\"type\":\"ping\"}}' | nc {}\n", tcp_addr);

        let listener = TcpListener::bind(&tcp_addr).await?;

        loop {
            let (stream, addr) = listener.accept().await?;
            info!("New connection from {}", addr);
            let state = Arc::clone(&state);

            tokio::spawn(async move {
                match handle_connection(stream, state).await {
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
