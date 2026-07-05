// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Daemon ↔ TUI inter-process communication via shared state file.
//!
//! The daemon periodically writes a [`DaemonSnapshot`](crate::ipc::DaemonSnapshot) to disk (JSON).
//! The TUI reads this file on startup and during refresh cycles to display
//! the daemon's live world model, episodic history, and active concerns.
//!
//! Default path: `$XDG_DATA_HOME/nix-mind/daemon_state.json`
//! (overridable via `NIX_MIND_STATE_DIR`).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

/// Errors that can occur during IPC operations.
#[derive(Debug)]
pub enum IpcError {
    /// Filesystem I/O error (create dir, write, rename).
    Io(std::io::Error),
    /// JSON serialization error.
    Serialize(serde_json::Error),
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IPC I/O error: {e}"),
            Self::Serialize(e) => write!(f, "IPC serialization error: {e}"),
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Serialize(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for IpcError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for IpcError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialize(e)
    }
}

/// Current IPC schema version. Increment when adding/removing fields.
pub const SNAPSHOT_VERSION: u32 = 3;

/// Snapshot of daemon cognitive state, serialized to disk for TUI consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonSnapshot {
    /// Schema version for forward/backward compatibility.
    #[serde(default = "default_snapshot_version")]
    pub version: u32,
    /// Seconds since Unix epoch when this snapshot was written.
    pub timestamp: u64,
    /// Total observations processed.
    pub observation_count: u64,
    /// Total anomalies detected.
    pub anomaly_count: u64,
    /// Prediction errors at each hierarchy level (Sensory, Features, Concepts, Goals).
    pub hierarchy_errors: [f64; 4],
    /// Total free energy from the predictive hierarchy.
    pub free_energy: f64,
    /// Whether the system is currently in a surprised state.
    pub is_surprised: bool,
    /// HDC world model drift similarity (1.0 = stable, 0.0 = total drift).
    pub drift_similarity: f32,
    /// Number of edges in the causal graph.
    pub causal_edge_count: usize,
    /// Number of episodes in episodic memory.
    pub episodic_count: usize,
    /// Active concerns in working memory.
    pub concerns: Vec<ConcernEntry>,
    /// Recent anomalies (last 20).
    pub recent_anomalies: Vec<AnomalyEntry>,
    /// Whether the daemon is currently running.
    pub daemon_running: bool,
    /// Daemon PID for liveness check.
    pub daemon_pid: u32,
    /// Summary of latest support assessment status.
    #[serde(default)]
    pub support_status: Option<String>,
    /// Number of active support recommendations.
    #[serde(default)]
    pub recommendation_count: usize,
    /// Predictive alerts from the LTC monitor.
    #[serde(default)]
    pub alerts: Vec<AlertEntry>,
    /// Top causal graph edges by confidence (for CausalExplorer widget).
    #[serde(default)]
    pub top_causal_edges: Vec<CausalEdgeEntry>,
    /// Memory usage percentage (from hardware probe).
    #[serde(default)]
    pub memory_used_percent: Option<f64>,
    /// Last watchdog verdict (written by `nix-mind watch`, read by daemon/TUI).
    #[serde(default)]
    pub watchdog_status: Option<String>,
    /// Whether the daemon is in degraded mode (hardware probe failures cached).
    #[serde(default)]
    pub degraded: bool,
    /// Rolling mean absolute error of predictions (lower = more accurate).
    #[serde(default)]
    pub prediction_accuracy: Option<f64>,
    /// Number of active inference maintenance plans generated (dry-run).
    #[serde(default)]
    pub maintenance_plan_count: u32,
    /// CPU load average [1min] from the latest hardware probe.
    #[serde(default)]
    pub load_average_1m: Option<f64>,
    /// Swap usage percentage.
    #[serde(default)]
    pub swap_used_percent: Option<f64>,
    /// Rolling EMA of recent anomaly scores (0 = calm, 1 = turbulent).
    /// Drives the adaptive allostatic threshold each cycle.
    #[serde(default)]
    pub anomaly_volatility_ema: f64,
    /// Active anomaly planning threshold this cycle (allostatic setpoint).
    /// Anomalies with score above this value triggered a plan.
    #[serde(default)]
    pub active_anomaly_threshold: f64,
    /// Expected free energy of the most recently recommended action (lower = more preferred).
    /// None if no plan was generated this cycle.
    #[serde(default)]
    pub last_plan_efe: Option<f64>,
}

/// A causal edge entry for IPC (lightweight copy of CausalEdge).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdgeEntry {
    pub from: String,
    pub to: String,
    pub confidence: f64,
}

impl fmt::Display for CausalEdgeEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} → {} ({:.2})", self.from, self.to, self.confidence)
    }
}

/// A concern tracked in working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcernEntry {
    /// Human-readable label.
    pub label: String,
    /// Activation level (0.0–1.0).
    pub activation: f64,
    /// Source category.
    pub source: String,
}

impl fmt::Display for ConcernEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{:.2}] ({})",
            self.label, self.activation, self.source
        )
    }
}

/// A recent journal anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEntry {
    /// Anomaly score (0.0–1.0).
    pub score: f64,
    /// Human-readable reason.
    pub reason: String,
    /// Originating systemd unit.
    pub unit: String,
    /// Diagnosed error type (e.g., "configuration_error", "dependency_failure").
    #[serde(default)]
    pub error_type: Option<String>,
    /// Suggested remediation from domain plugin diagnosis.
    #[serde(default)]
    pub suggestion: Option<String>,
}

impl fmt::Display for AnomalyEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({:.2})", self.unit, self.reason, self.score)
    }
}

/// Severity level for predictive alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "critical"),
            Self::Warning => write!(f, "warning"),
            Self::Info => write!(f, "info"),
        }
    }
}

/// A predictive alert entry from the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEntry {
    /// Metric name (e.g., "disk_usage", "memory_usage").
    pub metric: String,
    /// Current observed value.
    pub current_value: f64,
    /// Predicted future value.
    pub predicted_value: f64,
    /// Hours ahead for the prediction.
    pub hours_ahead: f64,
    /// Threshold that triggers the alert.
    pub threshold: f64,
    /// Confidence in the prediction (0.0-1.0).
    pub confidence: f64,
    /// Recommended action to mitigate.
    pub recommended_action: Option<String>,
    /// Severity of the alert.
    pub severity: AlertSeverity,
    /// Unix timestamp when first detected.
    pub first_seen: u64,
    /// Unix timestamp of last update.
    pub last_seen: u64,
    /// Number of consecutive detection cycles.
    pub consecutive_cycles: u32,
    /// Previous predicted value (for trend).
    pub prev_predicted_value: Option<f64>,
    /// Corroborating journal context entries.
    #[serde(default)]
    pub journal_context: Vec<String>,
}

impl DaemonSnapshot {
    /// Write this snapshot atomically to the given path.
    ///
    /// Uses write-to-temp + rename for atomicity, preventing the TUI
    /// from reading a partially-written file.
    pub fn write_to(&self, path: &Path) -> Result<(), IpcError> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)?;

        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, json.as_bytes())?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Read a snapshot from disk. Returns None if the file doesn't exist
    /// or can't be parsed.
    pub fn read_from(path: &Path) -> Option<Self> {
        let json = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&json).ok()
    }

    /// Check if this snapshot is reasonably fresh (written within `max_age_secs`).
    pub fn is_fresh(&self, max_age_secs: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        now.saturating_sub(self.timestamp) < max_age_secs
    }

    /// Check if the daemon process is still alive (via /proc on Linux).
    pub fn daemon_alive(&self) -> bool {
        Path::new(&format!("/proc/{}", self.daemon_pid)).exists()
    }

    /// A fully-populated snapshot with sensible defaults for use in tests.
    ///
    /// Override specific fields with struct update syntax:
    /// `DaemonSnapshot { anomaly_count: 99, ..DaemonSnapshot::test_default() }`
    pub fn test_default() -> Self {
        Self {
            version: SNAPSHOT_VERSION,
            timestamp: 1700000000,
            observation_count: 42,
            anomaly_count: 3,
            hierarchy_errors: [0.1, 0.2, 0.15, 0.05],
            free_energy: 0.35,
            is_surprised: false,
            drift_similarity: 0.95,
            causal_edge_count: 210,
            episodic_count: 5,
            concerns: vec![],
            recent_anomalies: vec![],
            daemon_running: true,
            daemon_pid: 12345,
            support_status: Some("Healthy".into()),
            recommendation_count: 0,
            alerts: vec![],
            top_causal_edges: vec![],
            memory_used_percent: Some(55.0),
            watchdog_status: None,
            degraded: false,
            prediction_accuracy: Some(3.0),
            maintenance_plan_count: 0,
            load_average_1m: Some(0.5),
            swap_used_percent: Some(10.0),
        }
    }
}

/// Daemon configuration — loadable from a JSON config file.
///
/// Default location: `$XDG_CONFIG_HOME/nix-mind/daemon.json`
/// (overridable via `NIX_MIND_CONFIG` environment variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// How often to take a full system snapshot (seconds).
    #[serde(default = "default_snapshot_interval")]
    pub snapshot_interval: u64,
    /// How often to check for high-frequency events like journal entries (seconds).
    #[serde(default = "default_poll_interval")]
    pub poll_interval: u64,
    /// Prediction error threshold for episodic storage.
    #[serde(default = "default_surprise_threshold")]
    pub surprise_threshold: f64,
    /// Maximum journal entries to process per poll cycle.
    #[serde(default = "default_journal_batch_size")]
    pub journal_batch_size: usize,
    /// IPC snapshot write frequency (every N poll cycles).
    #[serde(default = "default_ipc_write_interval")]
    pub ipc_write_interval: u64,
    /// Hebbian learning rate for causal graph.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// Ollama API endpoint (default: http://localhost:11434).
    #[serde(default = "default_ollama_endpoint")]
    pub ollama_endpoint: String,
    /// Primary Ollama model to use.
    #[serde(default = "default_ollama_model")]
    pub ollama_model: String,
    /// Ollama request timeout in seconds.
    #[serde(default = "default_ollama_timeout")]
    pub ollama_timeout: u64,
    /// Enable dynamic knowledge learning from resolved incidents.
    #[serde(default = "default_enable_knowledge_learning")]
    pub enable_knowledge_learning: bool,
    /// Port for the Prometheus metrics endpoint (observability feature).
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

fn default_snapshot_version() -> u32 {
    1 // Old snapshots without version field default to v1
}

fn default_snapshot_interval() -> u64 {
    60
}
fn default_poll_interval() -> u64 {
    5
}
fn default_surprise_threshold() -> f64 {
    0.3
}
fn default_journal_batch_size() -> usize {
    50
}
fn default_ipc_write_interval() -> u64 {
    10
}
fn default_learning_rate() -> f64 {
    0.1
}
fn default_ollama_endpoint() -> String {
    "http://localhost:11434".into()
}
fn default_ollama_model() -> String {
    "gemma3:1b".into()
}
fn default_ollama_timeout() -> u64 {
    30
}
fn default_enable_knowledge_learning() -> bool {
    true
}
fn default_metrics_port() -> u16 {
    9090
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: default_snapshot_interval(),
            poll_interval: default_poll_interval(),
            surprise_threshold: default_surprise_threshold(),
            journal_batch_size: default_journal_batch_size(),
            ipc_write_interval: default_ipc_write_interval(),
            learning_rate: default_learning_rate(),
            ollama_endpoint: default_ollama_endpoint(),
            ollama_model: default_ollama_model(),
            ollama_timeout: default_ollama_timeout(),
            enable_knowledge_learning: default_enable_knowledge_learning(),
            metrics_port: default_metrics_port(),
        }
    }
}

impl DaemonConfig {
    /// Load config from a JSON file. Returns default if file doesn't exist.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!(
                        "nix-mind: failed to parse config {}: {}, using defaults",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Load config from the default path, with environment override.
    pub fn load_default() -> Self {
        Self::load(&default_config_path())
    }

    /// Save config to a JSON file.
    pub fn save(&self, path: &Path) -> Result<(), IpcError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Default path for the daemon config file.
pub fn default_config_path() -> PathBuf {
    if let Some(config_path) = std::env::var_os("NIX_MIND_CONFIG") {
        PathBuf::from(config_path)
    } else if let Some(config_dir) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(config_dir)
            .join("nix-mind")
            .join("daemon.json")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config/nix-mind/daemon.json")
    } else {
        PathBuf::from("/etc/nix-mind/daemon.json")
    }
}

/// Default path for the daemon snapshot file.
pub fn default_snapshot_path() -> PathBuf {
    if let Some(state_dir) = std::env::var_os("NIX_MIND_STATE_DIR") {
        PathBuf::from(state_dir).join("daemon_state.json")
    } else if let Some(data_dir) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(data_dir)
            .join("nix-mind")
            .join("daemon_state.json")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home)
            .join(".local/share/nix-mind")
            .join("daemon_state.json")
    } else {
        PathBuf::from("/var/lib/nix-mind/daemon_state.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> DaemonSnapshot {
        DaemonSnapshot {
            concerns: vec![ConcernEntry {
                label: "high memory usage".into(),
                activation: 0.8,
                source: "system".into(),
            }],
            recent_anomalies: vec![AnomalyEntry {
                score: 0.7,
                reason: "OOM killer invoked".into(),
                unit: "kernel".into(),
                error_type: None,
                suggestion: None,
            }],
            support_status: None,
            memory_used_percent: None,
            prediction_accuracy: None,
            load_average_1m: None,
            swap_used_percent: None,
            ..DaemonSnapshot::test_default()
        }
    }

    #[test]
    fn test_serialize_deserialize() {
        let snap = sample_snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let restored: DaemonSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.observation_count, 42);
        assert_eq!(restored.hierarchy_errors, [0.1, 0.2, 0.15, 0.05]);
        assert_eq!(restored.concerns.len(), 1);
        assert_eq!(restored.recent_anomalies.len(), 1);
    }

    #[test]
    fn test_write_and_read() {
        let snap = sample_snapshot();
        let dir = std::env::temp_dir().join("nix-mind-ipc-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("daemon_state.json");

        snap.write_to(&path).unwrap();
        let restored = DaemonSnapshot::read_from(&path).unwrap();
        assert_eq!(restored.observation_count, 42);
        assert!(!restored.is_surprised);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_read_nonexistent() {
        let result = DaemonSnapshot::read_from(Path::new("/tmp/nonexistent-nix-mind.json"));
        assert!(result.is_none());
    }

    #[test]
    fn test_is_fresh() {
        let mut snap = sample_snapshot();
        // Old timestamp
        snap.timestamp = 1000;
        assert!(!snap.is_fresh(60));

        // Current timestamp
        snap.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(snap.is_fresh(60));
    }

    #[test]
    fn test_default_snapshot_path() {
        let path = default_snapshot_path();
        assert!(path.to_string_lossy().contains("daemon_state.json"));
    }

    #[test]
    fn test_config_default() {
        let config = DaemonConfig::default();
        assert_eq!(config.snapshot_interval, 60);
        assert_eq!(config.poll_interval, 5);
        assert!((config.surprise_threshold - 0.3).abs() < 1e-6);
        assert_eq!(config.journal_batch_size, 50);
        assert_eq!(config.ipc_write_interval, 10);
        assert!((config.learning_rate - 0.1).abs() < 1e-6);
        assert_eq!(config.ollama_endpoint, "http://localhost:11434");
        assert_eq!(config.ollama_model, "gemma3:1b");
        assert_eq!(config.ollama_timeout, 30);
        assert!(config.enable_knowledge_learning);
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = DaemonConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: DaemonConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.snapshot_interval, config.snapshot_interval);
        assert_eq!(restored.poll_interval, config.poll_interval);
    }

    #[test]
    fn test_config_partial_json() {
        // Only override some fields — rest should use defaults
        let json = r#"{"snapshot_interval": 120, "poll_interval": 10}"#;
        let config: DaemonConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.snapshot_interval, 120);
        assert_eq!(config.poll_interval, 10);
        assert!(
            (config.surprise_threshold - 0.3).abs() < 1e-6,
            "Should use default"
        );
        assert_eq!(config.journal_batch_size, 50, "Should use default");
    }

    #[test]
    fn test_config_save_and_load() {
        let dir = std::env::temp_dir().join("nix-mind-config-test");
        let path = dir.join("daemon.json");

        let mut config = DaemonConfig::default();
        config.snapshot_interval = 30;
        config.surprise_threshold = 0.5;

        config.save(&path).unwrap();
        let loaded = DaemonConfig::load(&path);
        assert_eq!(loaded.snapshot_interval, 30);
        assert!((loaded.surprise_threshold - 0.5).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_config_load_nonexistent() {
        let config = DaemonConfig::load(Path::new("/tmp/nonexistent-nix-mind-config.json"));
        // Should return defaults
        assert_eq!(config.snapshot_interval, 60);
    }

    #[test]
    fn test_config_load_invalid_json() {
        let dir = std::env::temp_dir().join("nix-mind-config-invalid");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("daemon.json");
        std::fs::write(&path, "not valid json!!!").unwrap();

        let config = DaemonConfig::load(&path);
        // Should return defaults on parse error
        assert_eq!(config.snapshot_interval, 60);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_default_config_path() {
        let path = default_config_path();
        assert!(path.to_string_lossy().contains("daemon.json"));
    }

    #[test]
    fn test_snapshot_version_present() {
        let snap = sample_snapshot();
        assert_eq!(snap.version, SNAPSHOT_VERSION);
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"version\""));
        let restored: DaemonSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.version, SNAPSHOT_VERSION);
    }

    #[test]
    fn test_ipc_error_display() {
        let io_err = IpcError::Io(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "access denied",
        ));
        let display = format!("{io_err}");
        assert!(display.contains("I/O error"));
        assert!(display.contains("access denied"));

        let json_err = serde_json::from_str::<DaemonSnapshot>("bad json").unwrap_err();
        let ser_err = IpcError::Serialize(json_err);
        let display = format!("{ser_err}");
        assert!(display.contains("serialization error"));
    }

    #[test]
    fn test_ipc_error_source() {
        use std::error::Error;
        let io_err = IpcError::from(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(io_err.source().is_some());
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(format!("{}", AlertSeverity::Critical), "critical");
        assert_eq!(format!("{}", AlertSeverity::Warning), "warning");
        assert_eq!(format!("{}", AlertSeverity::Info), "info");
    }

    #[test]
    fn test_causal_edge_entry_display() {
        let edge = CausalEdgeEntry {
            from: "nginx".into(),
            to: "port_80".into(),
            confidence: 0.875,
        };
        let display = format!("{edge}");
        assert!(display.contains("nginx"));
        assert!(display.contains("port_80"));
        assert!(display.contains("0.88") || display.contains("0.87"));
    }

    #[test]
    fn test_concern_entry_display() {
        let concern = ConcernEntry {
            label: "high memory".into(),
            activation: 0.8,
            source: "system".into(),
        };
        let display = format!("{concern}");
        assert!(display.contains("high memory"));
        assert!(display.contains("0.80"));
        assert!(display.contains("system"));
    }

    #[test]
    fn test_anomaly_entry_display() {
        let anomaly = AnomalyEntry {
            score: 0.7,
            reason: "OOM killer".into(),
            unit: "kernel".into(),
            error_type: None,
            suggestion: None,
        };
        let display = format!("{anomaly}");
        assert!(display.contains("kernel"));
        assert!(display.contains("OOM killer"));
        assert!(display.contains("0.70"));
    }

    #[test]
    fn test_write_to_bad_path() {
        let snap = sample_snapshot();
        let result = snap.write_to(Path::new("/nonexistent/deeply/nested/state.json"));
        // Should fail with IpcError::Io
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, IpcError::Io(_)));
    }

    #[test]
    fn test_old_snapshot_defaults_version_1() {
        let old_json = r#"{
            "timestamp": 1700000000,
            "observation_count": 1,
            "anomaly_count": 0,
            "hierarchy_errors": [0.0, 0.0, 0.0, 0.0],
            "free_energy": 0.0,
            "is_surprised": false,
            "drift_similarity": 1.0,
            "causal_edge_count": 0,
            "episodic_count": 0,
            "concerns": [],
            "recent_anomalies": [],
            "daemon_running": true,
            "daemon_pid": 1
        }"#;
        let snap: DaemonSnapshot = serde_json::from_str(old_json).unwrap();
        assert_eq!(
            snap.version, 1,
            "Old snapshots without version field should default to 1"
        );
        assert!(
            !snap.degraded,
            "Old snapshots should default to not degraded"
        );
    }
}
