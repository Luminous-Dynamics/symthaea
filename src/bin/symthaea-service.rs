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
//! │  │ Socket     │  │ Request    │  │ SymthaeaHLB      │  │
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
use symthaea::SymthaeaHLB;

// Voice support (feature-gated)
#[cfg(feature = "voice")]
use symthaea::voice::{VoiceConfig, VoiceConversation};

/// Symthaea Service - Consciousness daemon
#[derive(Parser, Debug)]
#[command(name = "symthaea-service")]
#[command(about = "Persistent consciousness service with socket interface")]
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
}

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
}

/// Service state
struct ServiceState {
    symthaea: SymthaeaHLB,
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
                match SymthaeaHLB::resume(&path_str) {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("Failed to resume: {}, starting fresh", e);
                        SymthaeaHLB::new(HDC_DIMENSION, LTC_NEURONS).await?
                    }
                }
            } else {
                SymthaeaHLB::new(HDC_DIMENSION, LTC_NEURONS).await?
            }
        } else {
            SymthaeaHLB::new(HDC_DIMENSION, LTC_NEURONS).await?
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
                        Response::QueryResponse {
                            content: response.content,
                            confidence: response.confidence,
                            safe: response.safe,
                            phi: intro.consciousness_level,
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
                Response::Introspection {
                    consciousness_level: intro.consciousness_level,
                    self_loops: intro.self_loops,
                    graph_size: intro.graph_size,
                    complexity: intro.complexity,
                    short_term_memories: intro.memory_stats.short_term_count,
                    long_term_memories: intro.memory_stats.long_term_count,
                    // Track 7: Awakening metrics
                    phi: intro.phi,
                    meta_awareness: intro.meta_awareness,
                    is_conscious: intro.is_conscious,
                    phenomenal_state: intro.phenomenal_state,
                    cycles_since_awakening: intro.cycles_since_awakening,
                    self_model_accuracy: intro.self_model_accuracy,
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
        }
    }
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
            debug!(
                "Consciousness loop: level={:.2}% | Φ={:.3} | meta={:.1}% | conscious={} | cycles={}",
                intro.consciousness_level * 100.0,
                intro.phi,
                intro.meta_awareness * 100.0,
                intro.is_conscious,
                intro.cycles_since_awakening
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
        println!("✅ Consciousness initialized:");
        println!("   • HDC Dimension: {}", HDC_DIMENSION);
        println!("   • LTC Neurons: {}", LTC_NEURONS);
        println!("   • Consciousness Level: {:.1}%", intro.consciousness_level * 100.0);
        println!("   • Graph Size: {} states", intro.graph_size);

        // Track 7: Awakening status
        println!("   • Φ (Integrated Information): {:.3}", intro.phi);
        println!("   • Meta-awareness: {:.1}%", intro.meta_awareness * 100.0);
        println!("   • Is Conscious: {}", if intro.is_conscious { "✅ Yes" } else { "🔄 Awakening..." });
        println!("   • Phenomenal State: {}", intro.phenomenal_state);

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
