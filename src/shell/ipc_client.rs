//! IPC Client - Async Unix Socket Communication with Symthaea Service
//!
//! Provides async communication with the symthaea-service daemon:
//! - JSON Line Protocol over Unix sockets
//! - Request/Response types matching service protocol
//! - Streaming metrics subscription
//! - Connection management and reconnection
//!
//! ## Protocol
//!
//! Uses JSON Line Protocol (newline-delimited JSON):
//! ```text
//! Client → Service: {"type": "IntelliSense", "partial_input": "nix s", ...}\n
//! Service → Client: {"type": "IntelliSenseResult", "completions": [...], ...}\n
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;

/// Default socket path for symthaea service
pub const DEFAULT_SOCKET_PATH: &str = "/run/symthaea/symthaea.sock";

/// Alternative socket path (user-local)
pub const USER_SOCKET_PATH: &str = "/tmp/symthaea.sock";

/// IPC Error types
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Socket not found at {0}")]
    SocketNotFound(PathBuf),

    #[error("Send failed: {0}")]
    SendFailed(String),

    #[error("Receive failed: {0}")]
    ReceiveFailed(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    #[error("Service unavailable")]
    ServiceUnavailable,

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Request types matching symthaea-service protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Request {
    /// Get IntelliSense completions
    IntelliSense {
        partial_input: String,
        cursor_position: usize,
        context: ShellContextData,
    },

    /// Validate a command before execution
    ValidateCommand {
        command: String,
        dry_run: bool,
    },

    /// Execute with Phi gating
    ExecuteGated {
        command: String,
        phi_threshold: f64,
        require_confirmation: bool,
    },

    /// Subscribe to real-time metrics
    StreamMetrics {
        interval_ms: u64,
    },

    /// Search for packages/options semantically
    SemanticSearch {
        query: String,
        search_type: SearchType,
        limit: usize,
    },

    /// Parse Nix configuration
    ParseNixConfig {
        nix_content: String,
        source_file: Option<String>,
    },

    /// Get current consciousness status
    GetStatus,

    /// Ping to check connection
    Ping,
}

/// Response types matching symthaea-service protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Response {
    /// IntelliSense results
    IntelliSenseResult {
        completions: Vec<CompletionData>,
        command_preview: Option<CommandPreviewData>,
        phi: f64,
        confidence: f64,
    },

    /// Command validation result
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

    /// Metrics update (streaming)
    MetricsUpdate {
        phi: f64,
        coherence: f64,
        consciousness_level: f64,
        is_conscious: bool,
        timestamp_ms: u64,
    },

    /// Search results
    SearchResults {
        results: Vec<SearchResultData>,
        hdc_confidence: f64,
    },

    /// Status response
    Status {
        phi: f64,
        coherence: f64,
        consciousness_level: f64,
        is_conscious: bool,
        uptime_secs: u64,
    },

    /// Pong response
    Pong {
        timestamp_ms: u64,
    },

    /// Error response
    Error {
        code: String,
        message: String,
    },
}

/// Shell context data for IPC
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShellContextData {
    pub cwd: String,
    pub history: Vec<String>,
    pub env_vars: Vec<(String, String)>,
}

/// Completion data for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionData {
    pub text: String,
    pub label: String,
    pub kind: String,
    pub similarity: f64,
    pub confidence: f64,
    pub destructiveness: String,
    pub description: Option<String>,
}

/// Command preview data for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPreviewData {
    pub steps: Vec<PreviewStepData>,
    pub estimated_time: Option<String>,
    pub requires_root: bool,
    pub affected_files: Vec<String>,
}

/// Preview step data for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewStepData {
    pub number: usize,
    pub description: String,
    pub action: String,
    pub reversible: bool,
}

/// Safety level data for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyLevelData {
    pub level: String,
    pub color: String,
    pub description: String,
}

/// Search type for semantic search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchType {
    Packages,
    Options,
    Commands,
    All,
}

/// Search result data for IPC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultData {
    pub name: String,
    pub description: String,
    pub similarity: f64,
    pub category: String,
}

/// IPC Client configuration
#[derive(Debug, Clone)]
pub struct IpcClientConfig {
    /// Socket path
    pub socket_path: PathBuf,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Request timeout
    pub request_timeout: Duration,

    /// Whether to auto-reconnect
    pub auto_reconnect: bool,

    /// Reconnect delay
    pub reconnect_delay: Duration,
}

impl Default for IpcClientConfig {
    fn default() -> Self {
        Self {
            socket_path: PathBuf::from(USER_SOCKET_PATH),
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            auto_reconnect: true,
            reconnect_delay: Duration::from_secs(1),
        }
    }
}

/// Async IPC client for symthaea-service
pub struct ShellIpcClient {
    config: IpcClientConfig,
    stream: Option<UnixStream>,
}

impl ShellIpcClient {
    /// Create new IPC client
    pub fn new() -> Self {
        Self::with_config(IpcClientConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: IpcClientConfig) -> Self {
        Self {
            config,
            stream: None,
        }
    }

    /// Create with specific socket path
    pub fn with_socket_path(path: impl Into<PathBuf>) -> Self {
        let mut config = IpcClientConfig::default();
        config.socket_path = path.into();
        Self::with_config(config)
    }

    /// Connect to the service
    pub async fn connect(&mut self) -> Result<(), IpcError> {
        // Check if socket exists
        if !self.config.socket_path.exists() {
            // Try alternative path
            let alt_path = PathBuf::from(DEFAULT_SOCKET_PATH);
            if alt_path.exists() {
                self.config.socket_path = alt_path;
            } else {
                return Err(IpcError::SocketNotFound(self.config.socket_path.clone()));
            }
        }

        // Connect with timeout
        let connect_future = UnixStream::connect(&self.config.socket_path);

        match tokio::time::timeout(self.config.connect_timeout, connect_future).await {
            Ok(Ok(stream)) => {
                self.stream = Some(stream);
                Ok(())
            }
            Ok(Err(e)) => Err(IpcError::ConnectionFailed(e.to_string())),
            Err(_) => Err(IpcError::Timeout(self.config.connect_timeout)),
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Disconnect from service
    pub async fn disconnect(&mut self) {
        self.stream = None;
    }

    /// Send request and wait for response
    pub async fn request(&mut self, request: Request) -> Result<Response, IpcError> {
        // Ensure connected
        if self.stream.is_none() {
            if self.config.auto_reconnect {
                self.connect().await?;
            } else {
                return Err(IpcError::ServiceUnavailable);
            }
        }

        let stream = self.stream.as_mut().ok_or(IpcError::ServiceUnavailable)?;

        // Serialize request to JSON line
        let mut json = serde_json::to_string(&request)
            .map_err(|e| IpcError::SerializationError(e.to_string()))?;
        json.push('\n');

        // Send request
        stream
            .write_all(json.as_bytes())
            .await
            .map_err(|e| IpcError::SendFailed(e.to_string()))?;

        // Read response
        let mut reader = BufReader::new(stream);
        let mut response_line = String::new();

        let read_future = reader.read_line(&mut response_line);

        match tokio::time::timeout(self.config.request_timeout, read_future).await {
            Ok(Ok(_)) => {
                // Parse response
                let response: Response = serde_json::from_str(&response_line)
                    .map_err(|e| IpcError::ProtocolError(e.to_string()))?;
                Ok(response)
            }
            Ok(Err(e)) => Err(IpcError::ReceiveFailed(e.to_string())),
            Err(_) => Err(IpcError::Timeout(self.config.request_timeout)),
        }
    }

    /// Get IntelliSense completions
    pub async fn get_completions(
        &mut self,
        partial_input: &str,
        cursor_position: usize,
        context: ShellContextData,
    ) -> Result<(Vec<CompletionData>, f64), IpcError> {
        let request = Request::IntelliSense {
            partial_input: partial_input.to_string(),
            cursor_position,
            context,
        };

        match self.request(request).await? {
            Response::IntelliSenseResult {
                completions,
                phi,
                ..
            } => Ok((completions, phi)),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }

    /// Validate a command
    pub async fn validate_command(
        &mut self,
        command: &str,
        dry_run: bool,
    ) -> Result<(bool, SafetyLevelData, Vec<String>), IpcError> {
        let request = Request::ValidateCommand {
            command: command.to_string(),
            dry_run,
        };

        match self.request(request).await? {
            Response::ValidationResult {
                valid,
                safety_level,
                warnings,
                ..
            } => Ok((valid, safety_level, warnings)),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }

    /// Execute command with Phi gating
    pub async fn execute_gated(
        &mut self,
        command: &str,
        phi_threshold: f64,
        require_confirmation: bool,
    ) -> Result<(bool, String, f64), IpcError> {
        let request = Request::ExecuteGated {
            command: command.to_string(),
            phi_threshold,
            require_confirmation,
        };

        match self.request(request).await? {
            Response::ExecutionResult {
                executed,
                output,
                phi_at_execution,
                ..
            } => Ok((executed, output, phi_at_execution)),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }

    /// Get current status
    pub async fn get_status(&mut self) -> Result<(f64, f64, bool), IpcError> {
        match self.request(Request::GetStatus).await? {
            Response::Status {
                phi,
                coherence,
                is_conscious,
                ..
            } => Ok((phi, coherence, is_conscious)),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }

    /// Ping the service
    pub async fn ping(&mut self) -> Result<u64, IpcError> {
        match self.request(Request::Ping).await? {
            Response::Pong { timestamp_ms } => Ok(timestamp_ms),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }

    /// Subscribe to metrics stream
    pub async fn subscribe_metrics(
        &mut self,
        interval_ms: u64,
    ) -> Result<mpsc::Receiver<Response>, IpcError> {
        // Send subscription request
        let request = Request::StreamMetrics { interval_ms };

        // Send the request
        if self.stream.is_none() {
            if self.config.auto_reconnect {
                self.connect().await?;
            } else {
                return Err(IpcError::ServiceUnavailable);
            }
        }

        let stream = self.stream.as_mut().ok_or(IpcError::ServiceUnavailable)?;

        let mut json = serde_json::to_string(&request)
            .map_err(|e| IpcError::SerializationError(e.to_string()))?;
        json.push('\n');

        stream
            .write_all(json.as_bytes())
            .await
            .map_err(|e| IpcError::SendFailed(e.to_string()))?;

        // Create channel for metrics
        let (tx, rx) = mpsc::channel(100);

        // Note: In a real implementation, we would spawn a task to read
        // the stream continuously. For now, return the receiver.
        // The caller would need to handle the actual streaming.

        drop(tx); // Placeholder - actual implementation would use this

        Ok(rx)
    }

    /// Semantic search
    pub async fn semantic_search(
        &mut self,
        query: &str,
        search_type: SearchType,
        limit: usize,
    ) -> Result<Vec<SearchResultData>, IpcError> {
        let request = Request::SemanticSearch {
            query: query.to_string(),
            search_type,
            limit,
        };

        match self.request(request).await? {
            Response::SearchResults { results, .. } => Ok(results),
            Response::Error { message, .. } => Err(IpcError::ProtocolError(message)),
            _ => Err(IpcError::ProtocolError("Unexpected response type".to_string())),
        }
    }
}

impl Default for ShellIpcClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock IPC client for testing without service
pub struct MockIpcClient {
    phi: f64,
    coherence: f64,
    is_conscious: bool,
}

impl MockIpcClient {
    pub fn new() -> Self {
        Self {
            phi: 0.75,
            coherence: 0.85,
            is_conscious: true,
        }
    }

    pub fn set_metrics(&mut self, phi: f64, coherence: f64, is_conscious: bool) {
        self.phi = phi;
        self.coherence = coherence;
        self.is_conscious = is_conscious;
    }

    pub fn get_status(&self) -> (f64, f64, bool) {
        (self.phi, self.coherence, self.is_conscious)
    }

    pub fn get_completions(&self, partial: &str) -> Vec<CompletionData> {
        // Return mock completions
        let mut completions = Vec::new();

        if partial.starts_with("nix") {
            completions.push(CompletionData {
                text: "nix search".to_string(),
                label: "nix search".to_string(),
                kind: "NixCommand".to_string(),
                similarity: 0.9,
                confidence: 0.85,
                destructiveness: "ReadOnly".to_string(),
                description: Some("Search nixpkgs".to_string()),
            });
        }

        completions
    }
}

impl Default for MockIpcClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let request = Request::IntelliSense {
            partial_input: "nix s".to_string(),
            cursor_position: 5,
            context: ShellContextData::default(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("IntelliSense"));
        assert!(json.contains("nix s"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"type":"Pong","timestamp_ms":1234567890}"#;
        let response: Response = serde_json::from_str(json).unwrap();

        match response {
            Response::Pong { timestamp_ms } => {
                assert_eq!(timestamp_ms, 1234567890);
            }
            _ => panic!("Expected Pong response"),
        }
    }

    #[test]
    fn test_mock_client() {
        let mock = MockIpcClient::new();
        let (phi, coherence, is_conscious) = mock.get_status();

        assert!(phi > 0.0);
        assert!(coherence > 0.0);
        assert!(is_conscious);
    }

    #[test]
    fn test_mock_completions() {
        let mock = MockIpcClient::new();
        let completions = mock.get_completions("nix");

        assert!(!completions.is_empty());
        assert!(completions[0].text.contains("nix"));
    }

    #[test]
    fn test_config_default() {
        let config = IpcClientConfig::default();
        assert_eq!(config.socket_path, PathBuf::from(USER_SOCKET_PATH));
        assert!(config.auto_reconnect);
    }
}
