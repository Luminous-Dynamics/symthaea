//! Daemon ↔ TUI inter-process communication via shared state file.
//!
//! The daemon periodically writes a [`DaemonSnapshot`] to disk (JSON).
//! The TUI reads this file on startup and during refresh cycles to display
//! the daemon's live world model, episodic history, and active concerns.
//!
//! Default path: `$XDG_DATA_HOME/nix-mind/daemon_state.json`
//! (overridable via `NIX_MIND_STATE_DIR`).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Snapshot of daemon cognitive state, serialized to disk for TUI consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonSnapshot {
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

/// A recent journal anomaly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEntry {
    /// Anomaly score (0.0–1.0).
    pub score: f64,
    /// Human-readable reason.
    pub reason: String,
    /// Originating systemd unit.
    pub unit: String,
}

impl DaemonSnapshot {
    /// Write this snapshot atomically to the given path.
    ///
    /// Uses write-to-temp + rename for atomicity, preventing the TUI
    /// from reading a partially-written file.
    pub fn write_to(&self, path: &Path) -> Result<(), String> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create dir: {}", e))?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serialize: {}", e))?;

        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, json.as_bytes())
            .map_err(|e| format!("write tmp: {}", e))?;
        std::fs::rename(&tmp_path, path)
            .map_err(|e| format!("rename: {}", e))?;
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
}

/// Default path for the daemon snapshot file.
pub fn default_snapshot_path() -> PathBuf {
    if let Some(state_dir) = std::env::var_os("NIX_MIND_STATE_DIR") {
        PathBuf::from(state_dir).join("daemon_state.json")
    } else if let Some(data_dir) = std::env::var_os("XDG_DATA_HOME") {
        PathBuf::from(data_dir).join("nix-mind").join("daemon_state.json")
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
            timestamp: 1700000000,
            observation_count: 42,
            anomaly_count: 3,
            hierarchy_errors: [0.1, 0.2, 0.15, 0.05],
            free_energy: 0.35,
            is_surprised: false,
            drift_similarity: 0.95,
            causal_edge_count: 210,
            episodic_count: 5,
            concerns: vec![ConcernEntry {
                label: "high memory usage".into(),
                activation: 0.8,
                source: "system".into(),
            }],
            recent_anomalies: vec![AnomalyEntry {
                score: 0.7,
                reason: "OOM killer invoked".into(),
                unit: "kernel".into(),
            }],
            daemon_running: true,
            daemon_pid: 12345,
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
}
