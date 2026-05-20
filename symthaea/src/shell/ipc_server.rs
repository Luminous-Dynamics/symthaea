// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! IPC Server - Service-side connection handler for Shell TUI
//!
//! Provides Unix socket IPC server that:
//! - Accepts connections from symthaea-shell clients
//! - Streams real-time MetricsSnapshot updates
//! - Handles command execution with Phi gating
//! - Provides IntelliSense completions via HDC similarity
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    CognitiveLoopService                          │
//! │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │
//! │  │ CfC Network │  │ HDC Encoder│  │ Voice Bridge│                │
//! │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘                 │
//! │        │               │               │                         │
//! │        └───────────────┼───────────────┘                         │
//! │                        ▼                                         │
//! │              ┌─────────────────┐                                 │
//! │              │   IpcServer     │◄──── Unix Socket                │
//! │              │  (this module)  │                                 │
//! │              └────────┬────────┘                                 │
//! │                       │                                          │
//! │         ┌─────────────┼─────────────┐                           │
//! │         ▼             ▼             ▼                           │
//! │   ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
//! │   │ Client 1 │  │ Client 2 │  │ Client N │  (shells)            │
//! │   └──────────┘  └──────────┘  └──────────┘                      │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use anyhow::{Context, Result};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{broadcast, mpsc, RwLock};

use super::context::{Completion, CompletionKind};
use super::ipc_client::{CompletionItem, IpcRequest, IpcResponse, MetricsSnapshot};

const IPC_PROTOCOL_VERSION: u32 = 1;

// ═══════════════════════════════════════════════════════════════════════════════
// SERVER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// IPC server configuration
#[derive(Debug, Clone)]
pub struct IpcServerConfig {
    /// Socket path (created on bind)
    pub socket_path: PathBuf,
    /// Maximum concurrent clients
    pub max_clients: usize,
    /// Metrics push interval
    pub metrics_interval: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Buffer size for reading requests
    pub buffer_size: usize,
}

impl Default for IpcServerConfig {
    fn default() -> Self {
        // Default to XDG_RUNTIME_DIR or /tmp
        let socket_path = std::env::var("XDG_RUNTIME_DIR")
            .map(|dir| PathBuf::from(dir).join("symthaea/symthaea.sock"))
            .unwrap_or_else(|_| PathBuf::from("/tmp/symthaea.sock"));

        Self {
            socket_path,
            max_clients: 10,
            metrics_interval: Duration::from_millis(100),
            request_timeout: Duration::from_secs(5),
            buffer_size: 65536,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRICS PROVIDER TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for providing consciousness metrics to the IPC server
///
/// Implemented by CognitiveLoopService to expose its internal state.
pub trait MetricsProvider: Send + Sync {
    /// Get current metrics snapshot
    fn get_metrics(&self) -> MetricsSnapshot;

    /// Get current Phi value
    fn phi(&self) -> f64;

    /// Get current coherence
    fn coherence(&self) -> f64;

    /// Check if system is conscious
    fn is_conscious(&self) -> bool;

    /// Get current cognitive depth as string
    fn cognitive_depth(&self) -> String;

    /// Get current response strategy as string
    fn current_strategy(&self) -> String;

    /// Check if in flow state
    fn in_flow(&self) -> bool;

    /// Get uptime in seconds
    fn uptime_secs(&self) -> u64;

    /// Get total cycles processed
    fn total_cycles(&self) -> u64;
}

/// Trait for executing commands with Phi gating
pub trait CommandExecutor: Send + Sync {
    /// Execute a command (with Phi check)
    fn execute(&self, command: &str, require_phi: Option<f64>) -> ExecutionResult;

    /// Validate a command without executing
    fn validate(&self, command: &str) -> ValidationResult;

    /// Get completions for input
    fn get_completions(&self, input: &str, cursor_pos: usize) -> Vec<Completion>;
}

/// Result of command execution
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub output: String,
    pub phi_at_execution: f64,
    pub vetoed: bool,
    pub veto_reason: Option<String>,
}

/// Result of command validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub safety_level: String,
    pub preview: Option<String>,
    pub warnings: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// IPC SERVER
// ═══════════════════════════════════════════════════════════════════════════════

/// IPC server state
pub struct IpcServer {
    /// Configuration
    config: IpcServerConfig,
    /// Metrics provider (cognitive loop service)
    metrics_provider: Arc<dyn MetricsProvider>,
    /// Command executor
    executor: Arc<dyn CommandExecutor>,
    /// Active client count
    active_clients: Arc<RwLock<usize>>,
    /// Shutdown signal sender
    shutdown_tx: Option<broadcast::Sender<()>>,
    /// Server start time (reserved for uptime reporting)
    _start_time: Instant,
}

impl IpcServer {
    /// Create a new IPC server
    pub fn new(
        config: IpcServerConfig,
        metrics_provider: Arc<dyn MetricsProvider>,
        executor: Arc<dyn CommandExecutor>,
    ) -> Self {
        Self {
            config,
            metrics_provider,
            executor,
            active_clients: Arc::new(RwLock::new(0)),
            shutdown_tx: None,
            _start_time: Instant::now(),
        }
    }

    /// Get socket path
    pub fn socket_path(&self) -> &Path {
        &self.config.socket_path
    }

    /// Get active client count
    pub async fn active_clients(&self) -> usize {
        *self.active_clients.read().await
    }

    /// Start the server (blocking)
    pub async fn run(&mut self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.config.socket_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .context("Failed to create socket directory")?;
        }

        // Remove existing socket if present
        if self.config.socket_path.exists() {
            tokio::fs::remove_file(&self.config.socket_path)
                .await
                .context("Failed to remove existing socket")?;
        }

        // Bind to socket
        let listener =
            UnixListener::bind(&self.config.socket_path).context("Failed to bind to socket")?;

        #[cfg(unix)]
        {
            let perms = std::fs::Permissions::from_mode(0o600);
            tokio::fs::set_permissions(&self.config.socket_path, perms)
                .await
                .context("Failed to set socket permissions")?;
        }

        tracing::info!(
            socket = %self.config.socket_path.display(),
            "IPC server listening"
        );

        // Setup shutdown channel
        let (shutdown_tx, _) = broadcast::channel(1);
        self.shutdown_tx = Some(shutdown_tx.clone());

        // Accept connections
        loop {
            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let clients = *self.active_clients.read().await;
                            if clients >= self.config.max_clients {
                                tracing::warn!("Max clients reached, rejecting connection");
                                continue;
                            }

                            if let Err(err) = authorize_peer(&stream, &self.config.socket_path) {
                                tracing::warn!(error = %err, "Unauthorized IPC client");
                                continue;
                            }

                            // Spawn client handler
                            let metrics_provider = Arc::clone(&self.metrics_provider);
                            let executor = Arc::clone(&self.executor);
                            let active_clients = Arc::clone(&self.active_clients);
                            let config = self.config.clone();
                            let mut shutdown_rx = shutdown_tx.subscribe();

                            tokio::spawn(async move {
                                // Increment client count
                                {
                                    let mut count = active_clients.write().await;
                                    *count += 1;
                                }

                                // Handle client
                                if let Err(e) = handle_client(
                                    stream,
                                    metrics_provider,
                                    executor,
                                    config,
                                    &mut shutdown_rx,
                                ).await {
                                    tracing::debug!(error = %e, "Client handler error");
                                }

                                // Decrement client count
                                {
                                    let mut count = active_clients.write().await;
                                    *count = count.saturating_sub(1);
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Accept error");
                        }
                    }
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Shutdown signal received");
                    break;
                }
            }
        }

        // Cleanup
        if self.config.socket_path.exists() {
            let _ = tokio::fs::remove_file(&self.config.socket_path).await;
        }

        Ok(())
    }

    /// Request graceful shutdown
    pub fn shutdown(&self) {
        if let Some(ref tx) = self.shutdown_tx {
            let _ = tx.send(());
        }
    }
}

/// Handle a single client connection
async fn handle_client(
    stream: UnixStream,
    metrics_provider: Arc<dyn MetricsProvider>,
    executor: Arc<dyn CommandExecutor>,
    config: IpcServerConfig,
    shutdown_rx: &mut broadcast::Receiver<()>,
) -> Result<()> {
    let (reader, writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut writer = BufWriter::new(writer);

    // Channel for sending responses/metrics
    let (response_tx, mut response_rx) = mpsc::channel::<IpcResponse>(100);

    // Track if client is subscribed to metrics
    let subscribed = Arc::new(RwLock::new(false));
    let subscribed_clone = Arc::clone(&subscribed);

    // Metrics pusher task
    let metrics_provider_clone = Arc::clone(&metrics_provider);
    let response_tx_clone = response_tx.clone();
    let metrics_interval = config.metrics_interval;

    let metrics_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(metrics_interval);
        loop {
            interval.tick().await;

            if *subscribed_clone.read().await {
                let metrics = metrics_provider_clone.get_metrics();
                if response_tx_clone
                    .send(IpcResponse::Metrics(metrics))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    });

    let mut negotiated_version: Option<u32> = None;

    loop {
        tokio::select! {
            // Check for shutdown
            _ = shutdown_rx.recv() => {
                tracing::debug!("Client handler shutting down");
                break;
            }

            // Send queued responses
            Some(response) = response_rx.recv() => {
                let data = rmp_serde::to_vec(&response)?;
                let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();
                writer.write_all(&len).await?;
                writer.write_all(&data).await?;
                writer.flush().await?;
            }

            // Read incoming requests
            result = tokio::time::timeout(config.request_timeout, read_frame(&mut reader, config.buffer_size)) => {
                match result {
                    Ok(Ok(Some(request))) => {
                        if let IpcRequest::Hello { version } = request {
                            if version != IPC_PROTOCOL_VERSION {
                                let response = IpcResponse::Error(format!(
                                    "Protocol version mismatch (client {version}, server {IPC_PROTOCOL_VERSION})"
                                ));
                                let data = rmp_serde::to_vec(&response)?;
                                let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();
                                writer.write_all(&len).await?;
                                writer.write_all(&data).await?;
                                writer.flush().await?;
                                break;
                            }
                            negotiated_version = Some(version);
                            let response = IpcResponse::HelloAck { server_version: IPC_PROTOCOL_VERSION };
                            let data = rmp_serde::to_vec(&response)?;
                            let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();
                            writer.write_all(&len).await?;
                            writer.write_all(&data).await?;
                            writer.flush().await?;
                            continue;
                        }

                        if negotiated_version.is_none() {
                            let response = IpcResponse::Error("Protocol version not negotiated".to_string());
                            let data = rmp_serde::to_vec(&response)?;
                            let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();
                            writer.write_all(&len).await?;
                            writer.write_all(&data).await?;
                            writer.flush().await?;
                            continue;
                        }

                        let response = handle_request(
                            request,
                            &metrics_provider,
                            &executor,
                            &subscribed,
                        ).await;

                        // Send response
                        let data = rmp_serde::to_vec(&response)?;
                        let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();
                        writer.write_all(&len).await?;
                        writer.write_all(&data).await?;
                        writer.flush().await?;
                    }
                    Ok(Ok(None)) => {
                        tracing::debug!("Client disconnected");
                        break;
                    }
                    Ok(Err(e)) => {
                        tracing::debug!(error = %e, "Read error");
                        break;
                    }
                    Err(_) => {
                        tracing::debug!("Request timeout");
                        break;
                    }
                }
            }
        }
    }

    // Cleanup
    metrics_task.abort();
    Ok(())
}

/// Handle a single request
async fn handle_request(
    request: IpcRequest,
    metrics_provider: &Arc<dyn MetricsProvider>,
    executor: &Arc<dyn CommandExecutor>,
    subscribed: &Arc<RwLock<bool>>,
) -> IpcResponse {
    match request {
        IpcRequest::Hello { .. } => {
            IpcResponse::Error("Handshake must occur before requests".to_string())
        }
        IpcRequest::SubscribeMetrics => {
            *subscribed.write().await = true;
            IpcResponse::Subscribed
        }

        IpcRequest::UnsubscribeMetrics => {
            *subscribed.write().await = false;
            IpcResponse::Subscribed
        }

        IpcRequest::GetMetrics => IpcResponse::Metrics(metrics_provider.get_metrics()),

        IpcRequest::Ping => IpcResponse::Pong,

        IpcRequest::Execute {
            command,
            require_phi,
        } => {
            let result = executor.execute(&command, require_phi);
            IpcResponse::ExecutionResult {
                success: result.success,
                output: result.output,
                phi_at_execution: result.phi_at_execution,
                vetoed: result.vetoed,
                veto_reason: result.veto_reason,
            }
        }

        IpcRequest::Validate { command } => {
            let result = executor.validate(&command);
            IpcResponse::ValidationResult {
                valid: result.valid,
                safety_level: result.safety_level,
                preview: result.preview,
                warnings: result.warnings,
            }
        }

        IpcRequest::GetCompletions { input, cursor_pos } => {
            let completions = executor.get_completions(&input, cursor_pos);
            let items: Vec<CompletionItem> = completions
                .into_iter()
                .map(|c| CompletionItem {
                    text: c.text,
                    label: c.label,
                    kind: format!("{:?}", c.kind),
                    similarity: c.similarity,
                    docs: c.documentation,
                })
                .collect();
            IpcResponse::Completions(items)
        }
    }
}

#[cfg(unix)]
fn authorize_peer(stream: &UnixStream, socket_path: &Path) -> Result<()> {
    let creds = stream
        .peer_cred()
        .context("Failed to read peer credentials")?;
    let socket_uid = std::fs::metadata(socket_path)
        .context("Failed to read socket metadata")?
        .uid();

    if creds.uid() == socket_uid || creds.uid() == 0 {
        Ok(())
    } else {
        anyhow::bail!(
            "UID {} not authorized for socket owner {}",
            creds.uid(),
            socket_uid
        );
    }
}

#[cfg(not(unix))]
fn authorize_peer(_stream: &UnixStream, _socket_path: &Path) -> Result<()> {
    Ok(())
}

async fn read_frame(
    reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>,
    max_size: usize,
) -> Result<Option<IpcRequest>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Ok(None);
        }
        Err(e) => return Err(e.into()),
    }

    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > max_size {
        anyhow::bail!("Invalid IPC frame size: {len}");
    }

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    let request = rmp_serde::from_slice::<IpcRequest>(&buf)?;
    Ok(Some(request))
}

// ═══════════════════════════════════════════════════════════════════════════════
// STUB IMPLEMENTATIONS (for testing without full cognitive loop)
// ═══════════════════════════════════════════════════════════════════════════════

/// Stub metrics provider for testing
pub struct StubMetricsProvider {
    phi: f64,
    coherence: f64,
    start_time: Instant,
}

impl StubMetricsProvider {
    pub fn new() -> Self {
        Self {
            phi: 0.7,
            coherence: 0.8,
            start_time: Instant::now(),
        }
    }
}

impl Default for StubMetricsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsProvider for StubMetricsProvider {
    fn get_metrics(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            phi: self.phi,
            coherence: self.coherence,
            is_conscious: self.phi > 0.3,
            cognitive_depth: "Cortical".to_string(),
            strategy: "Supportive".to_string(),
            in_flow: false,
            prediction_error: 0.15,
            emotional_valence: 0.1,
            emotional_arousal: 0.5,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            uptime_secs: self.start_time.elapsed().as_secs(),
            total_cycles: 0,
            consciousness_level: (self.phi + self.coherence) / 2.0,
            latency_ms: 0,
            #[cfg(feature = "vision-manifold")]
            mental_movie: None,
            last_observed_frame: None,
            connected_peers: Vec::new(),
        }
    }

    fn phi(&self) -> f64 {
        self.phi
    }
    fn coherence(&self) -> f64 {
        self.coherence
    }
    fn is_conscious(&self) -> bool {
        self.phi > 0.3
    }
    fn cognitive_depth(&self) -> String {
        "Cortical".to_string()
    }
    fn current_strategy(&self) -> String {
        "Supportive".to_string()
    }
    fn in_flow(&self) -> bool {
        false
    }
    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
    fn total_cycles(&self) -> u64 {
        0
    }
}

/// Stub command executor for testing
pub struct StubCommandExecutor;

impl StubCommandExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StubCommandExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandExecutor for StubCommandExecutor {
    fn execute(&self, command: &str, require_phi: Option<f64>) -> ExecutionResult {
        let phi = 0.7;

        if let Some(required) = require_phi {
            if phi < required {
                return ExecutionResult {
                    success: false,
                    output: String::new(),
                    phi_at_execution: phi,
                    vetoed: true,
                    veto_reason: Some(format!("Phi {phi:.2} below required {required:.2}")),
                };
            }
        }

        ExecutionResult {
            success: true,
            output: format!("[stub] Would execute: {command}"),
            phi_at_execution: phi,
            vetoed: false,
            veto_reason: None,
        }
    }

    fn validate(&self, command: &str) -> ValidationResult {
        let is_destructive = command.contains("rm") || command.contains("delete");

        ValidationResult {
            valid: true,
            safety_level: if is_destructive {
                "Destructive"
            } else {
                "Safe"
            }
            .to_string(),
            preview: Some(format!("Would execute: {command}")),
            warnings: if is_destructive {
                vec!["This command may be destructive".to_string()]
            } else {
                Vec::new()
            },
        }
    }

    fn get_completions(&self, input: &str, _cursor_pos: usize) -> Vec<Completion> {
        // Simple stub completions
        let commands = ["install", "remove", "search", "rebuild", "switch"];
        commands
            .iter()
            .filter(|cmd| cmd.starts_with(input))
            .map(|cmd| Completion::new(*cmd, CompletionKind::Command))
            .collect()
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
    fn test_stub_metrics_provider() {
        let provider = StubMetricsProvider::new();
        let metrics = provider.get_metrics();
        assert!(metrics.phi > 0.0);
        assert!(metrics.is_conscious);
    }

    #[test]
    fn test_stub_executor() {
        let executor = StubCommandExecutor::new();

        // Test successful execution
        let result = executor.execute("install nginx", None);
        assert!(result.success);
        assert!(!result.vetoed);

        // Test vetoed execution
        let result = executor.execute("install nginx", Some(0.9));
        assert!(!result.success);
        assert!(result.vetoed);
    }

    #[test]
    fn test_stub_validate() {
        let executor = StubCommandExecutor::new();

        let result = executor.validate("search firefox");
        assert!(result.valid);
        assert_eq!(result.safety_level, "Safe");

        let result = executor.validate("rm -rf /");
        assert!(result.valid);
        assert_eq!(result.safety_level, "Destructive");
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_stub_completions() {
        let executor = StubCommandExecutor::new();
        let completions = executor.get_completions("inst", 4);
        assert!(completions.iter().any(|c| c.text == "install"));
    }

    #[test]
    fn test_config_default() {
        let config = IpcServerConfig::default();
        assert!(config.max_clients > 0);
        assert!(config.metrics_interval.as_millis() > 0);
    }

    #[tokio::test]
    async fn test_read_frame_roundtrip() {
        let (mut client, server) = match UnixStream::pair() {
            Ok(pair) => pair,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::PermissionDenied {
                    eprintln!("Skipping IPC server test: UnixStream::pair not permitted");
                    return;
                }
                panic!("UnixStream::pair failed: {err}");
            }
        };
        let request = IpcRequest::Ping;
        let data = rmp_serde::to_vec(&request).unwrap();
        let len = (data.len().min(u32::MAX as usize) as u32).to_le_bytes();

        if let Err(err) = client.write_all(&len).await {
            if err.kind() == std::io::ErrorKind::PermissionDenied {
                eprintln!("Skipping IPC server test: write not permitted");
                return;
            }
            panic!("write length failed: {err}");
        }
        if let Err(err) = client.write_all(&data).await {
            if err.kind() == std::io::ErrorKind::PermissionDenied {
                eprintln!("Skipping IPC server test: write not permitted");
                return;
            }
            panic!("write payload failed: {err}");
        }

        let (read_half, _write_half) = server.into_split();
        let mut reader = BufReader::new(read_half);
        let parsed = match read_frame(&mut reader, 1024).await {
            Ok(parsed) => parsed,
            Err(err) => {
                if err.to_string().contains("Operation not permitted") {
                    eprintln!("Skipping IPC server test: {err}");
                    return;
                }
                panic!("read_frame failed: {err}");
            }
        };
        let parsed = match parsed {
            Some(parsed) => parsed,
            None => panic!("Expected IPC frame, got None"),
        };
        assert!(matches!(parsed, IpcRequest::Ping));
    }
}
