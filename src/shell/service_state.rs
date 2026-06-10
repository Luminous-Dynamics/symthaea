// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Service State Management
//!
//! D2: Audit Logging - Track all executed commands with Phi values
//! D3: State Persistence - Save/restore consciousness state across restarts
//!
//! ## Audit Log Format
//!
//! ```text
//! 2024-01-02T10:30:15.123Z|EXECUTE|phi=0.85|coherence=0.92|command="install nginx"|result=SUCCESS
//! ```
//!
//! ## State File Format
//!
//! MessagePack-encoded `PersistentState` struct.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

// ============================================================================
// D2: Audit Logging
// ============================================================================

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp of the entry
    pub timestamp: DateTime<Utc>,
    /// Type of action
    pub action_type: AuditActionType,
    /// Phi value at time of action
    pub phi: f64,
    /// Coherence value at time of action
    pub coherence: f64,
    /// The command or action performed
    pub command: String,
    /// Result of the action
    pub result: AuditResult,
    /// Client identifier (if known)
    pub client_id: Option<String>,
    /// Request ID for correlation
    pub request_id: Option<String>,
    /// Additional metadata
    pub metadata: Option<String>,
}

/// Type of audited action
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditActionType {
    /// Command execution
    Execute,
    /// Command validation
    Validate,
    /// IntelliSense request
    IntelliSense,
    /// Connection event
    Connect,
    /// Disconnection event
    Disconnect,
    /// Configuration change
    ConfigChange,
    /// State restore
    StateRestore,
    /// State save
    StateSave,
}

impl AuditActionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Execute => "EXECUTE",
            Self::Validate => "VALIDATE",
            Self::IntelliSense => "INTELLISENSE",
            Self::Connect => "CONNECT",
            Self::Disconnect => "DISCONNECT",
            Self::ConfigChange => "CONFIG_CHANGE",
            Self::StateRestore => "STATE_RESTORE",
            Self::StateSave => "STATE_SAVE",
        }
    }
}

/// Result of an audited action
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuditResult {
    Success,
    Failure(String),
    Vetoed(String),
    Pending,
}

impl AuditResult {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Success => "SUCCESS",
            Self::Failure(_) => "FAILURE",
            Self::Vetoed(_) => "VETOED",
            Self::Pending => "PENDING",
        }
    }
}

/// Audit logger for tracking all actions
pub struct AuditLogger {
    /// Path to audit log file
    log_path: PathBuf,
    /// Log file writer
    writer: Option<BufWriter<File>>,
    /// Whether logging is enabled
    enabled: bool,
    /// Maximum log file size before rotation
    max_size_bytes: u64,
    /// Number of rotated files to keep
    rotation_count: usize,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(log_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let log_dir = log_dir.as_ref();
        fs::create_dir_all(log_dir)?;

        let log_path = log_dir.join("audit.log");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        Ok(Self {
            log_path,
            writer: Some(BufWriter::new(file)),
            enabled: true,
            max_size_bytes: 10 * 1024 * 1024, // 10MB
            rotation_count: 5,
        })
    }

    /// Disable logging
    pub fn disabled() -> Self {
        Self {
            log_path: PathBuf::new(),
            writer: None,
            enabled: false,
            max_size_bytes: 0,
            rotation_count: 0,
        }
    }

    /// Log an audit entry
    pub fn log(&mut self, entry: &AuditEntry) -> std::io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Check if rotation is needed
        if let Ok(metadata) = fs::metadata(&self.log_path) {
            if metadata.len() > self.max_size_bytes {
                self.rotate()?;
            }
        }

        // Format entry as pipe-delimited line
        let line = format!(
            "{}|{}|phi={:.4}|coherence={:.4}|command={:?}|result={}\n",
            entry.timestamp.to_rfc3339(),
            entry.action_type.as_str(),
            entry.phi,
            entry.coherence,
            entry.command,
            entry.result.as_str(),
        );

        if let Some(ref mut writer) = self.writer {
            writer.write_all(line.as_bytes())?;
            writer.flush()?;
        }

        Ok(())
    }

    /// Log a command execution
    pub fn log_execute(
        &mut self,
        command: &str,
        phi: f64,
        coherence: f64,
        result: AuditResult,
        request_id: Option<String>,
    ) -> std::io::Result<()> {
        self.log(&AuditEntry {
            timestamp: Utc::now(),
            action_type: AuditActionType::Execute,
            phi,
            coherence,
            command: command.to_string(),
            result,
            client_id: None,
            request_id,
            metadata: None,
        })
    }

    /// Rotate log files
    fn rotate(&mut self) -> std::io::Result<()> {
        // Close current writer
        self.writer = None;

        // Rotate files: audit.log.4 -> audit.log.5, etc.
        for i in (1..self.rotation_count).rev() {
            let from = self.log_path.with_extension(format!("log.{i}"));
            let to = self.log_path.with_extension(format!("log.{}", i + 1));
            if from.exists() {
                fs::rename(&from, &to)?;
            }
        }

        // Rename current log to .1
        if self.log_path.exists() {
            let rotated = self.log_path.with_extension("log.1");
            fs::rename(&self.log_path, &rotated)?;
        }

        // Open new log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        self.writer = Some(BufWriter::new(file));

        Ok(())
    }

    /// Get log file path
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }
}

// ============================================================================
// D3: State Persistence
// ============================================================================

/// Persistent state saved across restarts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentState {
    /// Version for forward compatibility
    pub version: u32,
    /// Last saved timestamp
    pub saved_at: DateTime<Utc>,
    /// Consciousness metrics
    pub consciousness: ConsciousnessState,
    /// Command history (last N commands)
    pub command_history: VecDeque<HistoricalCommand>,
    /// Session statistics
    pub session_stats: SessionStats,
    /// Custom user preferences
    pub preferences: UserPreferences,
}

/// Saved consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    pub phi: f64,
    pub coherence: f64,
    pub consciousness_level: f64,
    pub is_conscious: bool,
    /// Accumulated Phi over time (for trends)
    pub phi_history: VecDeque<f64>,
}

/// Historical command entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalCommand {
    pub command: String,
    pub timestamp: DateTime<Utc>,
    pub phi_at_execution: f64,
    pub success: bool,
}

/// Session statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionStats {
    /// Total commands executed
    pub commands_executed: u64,
    /// Commands vetoed by safety
    pub commands_vetoed: u64,
    /// Commands requiring confirmation
    pub confirmations_requested: u64,
    /// Total session time in seconds
    pub total_session_time_secs: u64,
    /// Number of reconnections
    pub reconnection_count: u64,
}

/// User preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Preferred Phi threshold for commands
    pub phi_threshold: f64,
    /// Enable audio notifications
    pub audio_enabled: bool,
    /// Enable desktop notifications
    pub notifications_enabled: bool,
    /// Color theme
    pub theme: String,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            phi_threshold: 0.5,
            audio_enabled: false,
            notifications_enabled: true,
            theme: "dark".to_string(),
        }
    }
}

impl Default for PersistentState {
    fn default() -> Self {
        Self {
            version: 1,
            saved_at: Utc::now(),
            consciousness: ConsciousnessState {
                phi: 0.5,
                coherence: 0.5,
                consciousness_level: 0.5,
                is_conscious: true,
                phi_history: VecDeque::new(),
            },
            command_history: VecDeque::new(),
            session_stats: SessionStats::default(),
            preferences: UserPreferences::default(),
        }
    }
}

/// State persistence manager
#[allow(dead_code)] // Fields reserved for auto-save feature
pub struct StateManager {
    /// Path to state file
    state_path: PathBuf,
    /// Current state
    state: PersistentState,
    /// Whether auto-save is enabled
    auto_save: bool,
    /// Auto-save interval in seconds
    auto_save_interval_secs: u64,
    /// Maximum history entries to keep
    max_history_entries: usize,
}

impl StateManager {
    /// Create a new state manager
    pub fn new(state_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let state_dir = state_dir.as_ref();
        fs::create_dir_all(state_dir)?;

        let state_path = state_dir.join("state.msgpack");

        // Try to load existing state
        let state = if state_path.exists() {
            match Self::load_state(&state_path) {
                Ok(state) => state,
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load state, using defaults");
                    PersistentState::default()
                }
            }
        } else {
            PersistentState::default()
        };

        Ok(Self {
            state_path,
            state,
            auto_save: true,
            auto_save_interval_secs: 60,
            max_history_entries: 1000,
        })
    }

    /// Load state from file
    fn load_state(path: &Path) -> std::io::Result<PersistentState> {
        let data = fs::read(path)?;
        rmp_serde::from_slice(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Save state to file
    pub fn save(&mut self) -> std::io::Result<()> {
        self.state.saved_at = Utc::now();

        let data = rmp_serde::to_vec(&self.state)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Write to temp file first, then rename (atomic)
        let temp_path = self.state_path.with_extension("msgpack.tmp");
        fs::write(&temp_path, &data)?;
        fs::rename(&temp_path, &self.state_path)?;

        tracing::debug!(path = %self.state_path.display(), "State saved");
        Ok(())
    }

    /// Get current state
    pub fn state(&self) -> &PersistentState {
        &self.state
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> &mut PersistentState {
        &mut self.state
    }

    /// Update consciousness metrics
    pub fn update_consciousness(&mut self, phi: f64, coherence: f64, is_conscious: bool) {
        self.state.consciousness.phi = phi;
        self.state.consciousness.coherence = coherence;
        self.state.consciousness.is_conscious = is_conscious;
        self.state.consciousness.consciousness_level = (phi + coherence) / 2.0;

        // Keep history bounded
        self.state.consciousness.phi_history.push_back(phi);
        if self.state.consciousness.phi_history.len() > 100 {
            self.state.consciousness.phi_history.pop_front();
        }
    }

    /// Add command to history
    pub fn add_to_history(&mut self, command: String, phi: f64, success: bool) {
        self.state.command_history.push_back(HistoricalCommand {
            command,
            timestamp: Utc::now(),
            phi_at_execution: phi,
            success,
        });

        // Keep history bounded
        while self.state.command_history.len() > self.max_history_entries {
            self.state.command_history.pop_front();
        }

        self.state.session_stats.commands_executed += 1;
    }

    /// Record a vetoed command
    pub fn record_veto(&mut self) {
        self.state.session_stats.commands_vetoed += 1;
    }

    /// Record a confirmation request
    pub fn record_confirmation(&mut self) {
        self.state.session_stats.confirmations_requested += 1;
    }

    /// Record a reconnection
    pub fn record_reconnection(&mut self) {
        self.state.session_stats.reconnection_count += 1;
    }

    /// Get state file path
    pub fn state_path(&self) -> &Path {
        &self.state_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_audit_logger() {
        let dir = tempdir().unwrap();
        let mut logger = AuditLogger::new(dir.path()).unwrap();

        let entry = AuditEntry {
            timestamp: Utc::now(),
            action_type: AuditActionType::Execute,
            phi: 0.85,
            coherence: 0.92,
            command: "install nginx".to_string(),
            result: AuditResult::Success,
            client_id: None,
            request_id: Some("abc123".to_string()),
            metadata: None,
        };

        logger.log(&entry).unwrap();

        // Check file was created and has content
        let content = fs::read_to_string(logger.log_path()).unwrap();
        assert!(content.contains("EXECUTE"));
        assert!(content.contains("install nginx"));
    }

    #[test]
    fn test_state_persistence() {
        let dir = tempdir().unwrap();
        let mut manager = StateManager::new(dir.path()).unwrap();

        // Update some state
        manager.update_consciousness(0.85, 0.92, true);
        manager.add_to_history("install nginx".to_string(), 0.85, true);

        // Save
        manager.save().unwrap();

        // Create a new manager and verify state was loaded
        let manager2 = StateManager::new(dir.path()).unwrap();
        assert!((manager2.state().consciousness.phi - 0.85).abs() < 0.01);
        assert_eq!(manager2.state().command_history.len(), 1);
    }
}
