// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learning Persistence: Save and Load Learned Patterns
//!
//! Persists the consciousness learning state across sessions:
//! - AdaptiveThresholds: Learned Φ and confidence boundaries
//! - OutcomePatterns: Success rates per quadrant/strategy
//!
//! Uses JSON for human-readable state that can be inspected and debugged.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Adaptive thresholds for consciousness-language integration.
/// Originally from consciousness_language_integration module; defined locally
/// since that module was removed during the crate consolidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    /// Current Φ threshold for engaging deeper processing
    pub phi_threshold: f64,
    /// Confidence threshold for accepting a prediction
    pub confidence_threshold: f64,
    /// Number of successful predictions
    pub successes: u64,
    /// Number of failed predictions
    pub failures: u64,
    /// History of threshold adjustments (timestamp, old, new)
    pub adjustment_history: Vec<(u64, f64, f64)>,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            phi_threshold: 0.5,
            confidence_threshold: 0.6,
            successes: 0,
            failures: 0,
            adjustment_history: Vec::new(),
        }
    }
}

/// Outcome patterns tracking success rates across different strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePatterns {
    /// Samples per quadrant (low-φ/low-conf, low-φ/high-conf, high-φ/low-conf, high-φ/high-conf)
    pub quadrant_samples: [u32; 4],
    /// Success rate per quadrant
    pub quadrant_success_rates: [f64; 4],
}

impl Default for OutcomePatterns {
    fn default() -> Self {
        Self {
            quadrant_samples: [0; 4],
            quadrant_success_rates: [0.0; 4],
        }
    }
}

// ============================================================================
// LEARNING STATE CONTAINER
// ============================================================================

/// Complete learning state that persists across sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningState {
    /// Schema version for compatibility
    pub version: u32,
    /// When this state was saved (Unix millis)
    pub saved_at: u64,
    /// Adaptive thresholds
    pub thresholds: AdaptiveThresholds,
    /// Outcome patterns
    pub patterns: OutcomePatterns,
    /// Total processing count when saved
    pub total_processed: u64,
    /// Session count (incremented each load)
    pub session_count: u64,
}

impl Default for LearningState {
    fn default() -> Self {
        Self {
            version: LEARNING_STATE_VERSION,
            saved_at: 0,
            thresholds: AdaptiveThresholds::default(),
            patterns: OutcomePatterns::default(),
            total_processed: 0,
            session_count: 0,
        }
    }
}

/// Current learning state schema version
pub const LEARNING_STATE_VERSION: u32 = 1;

// ============================================================================
// PERSISTENCE CONFIGURATION
// ============================================================================

/// Configuration for learning persistence
#[derive(Debug, Clone)]
pub struct LearningPersistenceConfig {
    /// Directory to store learning state
    pub data_dir: PathBuf,
    /// Filename for the state file
    pub filename: String,
    /// Auto-save interval (None = disabled)
    pub auto_save_interval: Option<std::time::Duration>,
    /// Keep N previous versions for rollback
    pub keep_versions: usize,
}

impl Default for LearningPersistenceConfig {
    fn default() -> Self {
        // Default to ~/.symthaea/learning/
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("symthaea")
            .join("learning");

        Self {
            data_dir,
            filename: "consciousness_learning.json".to_string(),
            auto_save_interval: Some(std::time::Duration::from_secs(300)), // 5 minutes
            keep_versions: 3,
        }
    }
}

impl LearningPersistenceConfig {
    /// Use a custom data directory
    pub fn with_dir(mut self, dir: PathBuf) -> Self {
        self.data_dir = dir;
        self
    }

    /// Set auto-save interval
    pub fn with_auto_save(mut self, interval: std::time::Duration) -> Self {
        self.auto_save_interval = Some(interval);
        self
    }

    /// Disable auto-save
    pub fn without_auto_save(mut self) -> Self {
        self.auto_save_interval = None;
        self
    }

    /// Get the full path to the state file
    pub fn state_path(&self) -> PathBuf {
        self.data_dir.join(&self.filename)
    }

    /// Get path for a versioned backup
    pub fn backup_path(&self, version: usize) -> PathBuf {
        self.data_dir
            .join(format!("{}.{}.bak", self.filename, version))
    }
}

// ============================================================================
// PERSISTENCE MANAGER
// ============================================================================

/// Manages persistence of learning state
pub struct LearningPersistence {
    config: LearningPersistenceConfig,
    /// Current state
    state: LearningState,
    /// Whether state has changed since last save
    dirty: bool,
    /// Last save time
    last_save: Option<std::time::Instant>,
}

impl LearningPersistence {
    /// Create a new persistence manager with default config
    pub fn new() -> Self {
        Self::with_config(LearningPersistenceConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LearningPersistenceConfig) -> Self {
        Self {
            config,
            state: LearningState::default(),
            dirty: false,
            last_save: None,
        }
    }

    /// Initialize: create directory and load existing state if present
    pub fn initialize(&mut self) -> Result<(), LearningPersistenceError> {
        // Ensure directory exists
        if !self.config.data_dir.exists() {
            fs::create_dir_all(&self.config.data_dir)
                .map_err(|e| LearningPersistenceError::Io(e.to_string()))?;
        }

        // Try to load existing state
        match self.load() {
            Ok(state) => {
                self.state = state;
                self.state.session_count += 1; // New session
                self.dirty = true; // Mark dirty to save updated session count
                Ok(())
            }
            Err(LearningPersistenceError::NotFound) => {
                // No existing state, use defaults
                self.state = LearningState::default();
                self.state.session_count = 1;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Load state from disk
    pub fn load(&self) -> Result<LearningState, LearningPersistenceError> {
        let path = self.config.state_path();

        if !path.exists() {
            return Err(LearningPersistenceError::NotFound);
        }

        let mut file =
            fs::File::open(&path).map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        let state: LearningState = serde_json::from_str(&contents)
            .map_err(|e| LearningPersistenceError::Deserialize(e.to_string()))?;

        // Version check
        if state.version > LEARNING_STATE_VERSION {
            return Err(LearningPersistenceError::VersionMismatch {
                file_version: state.version,
                current_version: LEARNING_STATE_VERSION,
            });
        }

        Ok(state)
    }

    /// Save current state to disk
    pub fn save(&mut self) -> Result<(), LearningPersistenceError> {
        // Rotate backups
        self.rotate_backups()?;

        // Update timestamp
        self.state.saved_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Serialize to JSON (pretty printed for readability)
        let json = serde_json::to_string_pretty(&self.state)
            .map_err(|e| LearningPersistenceError::Serialize(e.to_string()))?;

        // Write to file
        let path = self.config.state_path();
        let mut file =
            fs::File::create(&path).map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        file.write_all(json.as_bytes())
            .map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        self.dirty = false;
        self.last_save = Some(std::time::Instant::now());

        Ok(())
    }

    /// Rotate backup files
    fn rotate_backups(&self) -> Result<(), LearningPersistenceError> {
        let main_path = self.config.state_path();

        if !main_path.exists() {
            return Ok(());
        }

        // Rotate: .2.bak -> .3.bak, .1.bak -> .2.bak, etc.
        for i in (1..self.config.keep_versions).rev() {
            let old_path = self.config.backup_path(i);
            let new_path = self.config.backup_path(i + 1);
            if old_path.exists() {
                if let Err(e) = fs::rename(&old_path, &new_path) {
                    tracing::warn!(
                        from = %old_path.display(), to = %new_path.display(),
                        error = %e, "Backup rotation rename failed"
                    );
                }
            }
        }

        // Current -> .1.bak
        let first_backup = self.config.backup_path(1);
        if let Err(e) = fs::copy(&main_path, &first_backup) {
            tracing::warn!(
                from = %main_path.display(), to = %first_backup.display(),
                error = %e, "Backup copy failed"
            );
        }

        Ok(())
    }

    /// Check if auto-save is due
    pub fn should_auto_save(&self) -> bool {
        if !self.dirty {
            return false;
        }

        match (self.config.auto_save_interval, self.last_save) {
            (Some(interval), Some(last)) => last.elapsed() >= interval,
            (Some(_), None) => true, // Never saved, should save
            (None, _) => false,      // Auto-save disabled
        }
    }

    /// Auto-save if due (call periodically)
    pub fn maybe_auto_save(&mut self) -> Result<bool, LearningPersistenceError> {
        if self.should_auto_save() {
            self.save()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update thresholds and mark dirty
    pub fn update_thresholds(&mut self, thresholds: AdaptiveThresholds) {
        self.state.thresholds = thresholds;
        self.dirty = true;
    }

    /// Update patterns and mark dirty
    pub fn update_patterns(&mut self, patterns: OutcomePatterns) {
        self.state.patterns = patterns;
        self.dirty = true;
    }

    /// Update total processed count
    pub fn update_processed_count(&mut self, count: u64) {
        self.state.total_processed = count;
        self.dirty = true;
    }

    /// Get current thresholds
    pub fn thresholds(&self) -> &AdaptiveThresholds {
        &self.state.thresholds
    }

    /// Get current patterns
    pub fn patterns(&self) -> &OutcomePatterns {
        &self.state.patterns
    }

    /// Get the full state
    pub fn state(&self) -> &LearningState {
        &self.state
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get session count
    pub fn session_count(&self) -> u64 {
        self.state.session_count
    }

    /// Rollback to a previous version
    pub fn rollback(&mut self, version: usize) -> Result<(), LearningPersistenceError> {
        if version == 0 || version > self.config.keep_versions {
            return Err(LearningPersistenceError::InvalidVersion(version));
        }

        let backup_path = self.config.backup_path(version);
        if !backup_path.exists() {
            return Err(LearningPersistenceError::NotFound);
        }

        let mut file = fs::File::open(&backup_path)
            .map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| LearningPersistenceError::Io(e.to_string()))?;

        self.state = serde_json::from_str(&contents)
            .map_err(|e| LearningPersistenceError::Deserialize(e.to_string()))?;

        self.dirty = true;
        Ok(())
    }

    /// Get statistics about the learning state
    pub fn stats(&self) -> LearningStats {
        let accuracy = if self.state.thresholds.successes + self.state.thresholds.failures > 0 {
            self.state.thresholds.successes as f64
                / (self.state.thresholds.successes + self.state.thresholds.failures) as f64
        } else {
            0.0
        };

        let total_samples: u32 = self.state.patterns.quadrant_samples.iter().sum();

        LearningStats {
            session_count: self.state.session_count,
            total_processed: self.state.total_processed,
            successes: self.state.thresholds.successes,
            failures: self.state.thresholds.failures,
            accuracy,
            total_samples,
            phi_threshold: self.state.thresholds.phi_threshold,
            confidence_threshold: self.state.thresholds.confidence_threshold,
            adjustment_count: self.state.thresholds.adjustment_history.len(),
        }
    }
}

impl Default for LearningPersistence {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// STATISTICS
// ============================================================================

/// Statistics about learning state
#[derive(Debug, Clone)]
pub struct LearningStats {
    /// Number of sessions
    pub session_count: u64,
    /// Total inputs processed
    pub total_processed: u64,
    /// Successful predictions
    pub successes: u64,
    /// Failed predictions
    pub failures: u64,
    /// Current accuracy
    pub accuracy: f64,
    /// Total outcome samples
    pub total_samples: u32,
    /// Current Φ threshold
    pub phi_threshold: f64,
    /// Current confidence threshold
    pub confidence_threshold: f64,
    /// Number of threshold adjustments made
    pub adjustment_count: usize,
}

impl std::fmt::Display for LearningStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Learning Stats:\n\
             - Sessions: {}\n\
             - Processed: {}\n\
             - Accuracy: {:.1}% ({}/{} predictions)\n\
             - Outcome samples: {}\n\
             - Thresholds: Φ={:.2}, conf={:.2}\n\
             - Adjustments: {}",
            self.session_count,
            self.total_processed,
            self.accuracy * 100.0,
            self.successes,
            self.successes + self.failures,
            self.total_samples,
            self.phi_threshold,
            self.confidence_threshold,
            self.adjustment_count,
        )
    }
}

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during persistence operations
#[derive(Debug, Clone)]
pub enum LearningPersistenceError {
    /// File not found (no existing state)
    NotFound,
    /// I/O error
    Io(String),
    /// Serialization error
    Serialize(String),
    /// Deserialization error
    Deserialize(String),
    /// Version mismatch
    VersionMismatch {
        file_version: u32,
        current_version: u32,
    },
    /// Invalid rollback version
    InvalidVersion(usize),
}

impl std::fmt::Display for LearningPersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "Learning state file not found"),
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Serialize(e) => write!(f, "Serialization error: {}", e),
            Self::Deserialize(e) => write!(f, "Deserialization error: {}", e),
            Self::VersionMismatch {
                file_version,
                current_version,
            } => {
                write!(
                    f,
                    "Version mismatch: file v{} > current v{}",
                    file_version, current_version
                )
            }
            Self::InvalidVersion(v) => write!(f, "Invalid rollback version: {}", v),
        }
    }
}

impl std::error::Error for LearningPersistenceError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = LearningPersistenceConfig::default();
        assert!(config.data_dir.to_string_lossy().contains("symthaea"));
        assert_eq!(config.keep_versions, 3);
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let config = LearningPersistenceConfig::default().with_dir(dir.path().to_path_buf());

        let mut persist = LearningPersistence::with_config(config);
        persist.initialize().unwrap();

        // Modify state
        let mut thresholds = persist.thresholds().clone();
        thresholds.phi_threshold = 0.6;
        thresholds.successes = 100;
        persist.update_thresholds(thresholds);

        // Save
        persist.save().unwrap();

        // Load in new instance
        let config2 = LearningPersistenceConfig::default().with_dir(dir.path().to_path_buf());
        let mut persist2 = LearningPersistence::with_config(config2);
        persist2.initialize().unwrap();

        assert!((persist2.thresholds().phi_threshold - 0.6).abs() < 0.001);
        assert_eq!(persist2.thresholds().successes, 100);
        assert_eq!(persist2.session_count(), 2); // Second session
    }

    #[test]
    fn test_dirty_tracking() {
        let mut persist = LearningPersistence::new();
        assert!(!persist.is_dirty());

        let mut thresholds = persist.thresholds().clone();
        thresholds.phi_threshold = 0.7;
        persist.update_thresholds(thresholds);

        assert!(persist.is_dirty());
    }

    #[test]
    fn test_stats() {
        let mut persist = LearningPersistence::new();

        let mut thresholds = AdaptiveThresholds::default();
        thresholds.successes = 80;
        thresholds.failures = 20;
        persist.update_thresholds(thresholds);

        let stats = persist.stats();
        assert!((stats.accuracy - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_rollback() {
        let dir = tempdir().unwrap();
        let config = LearningPersistenceConfig::default().with_dir(dir.path().to_path_buf());

        let mut persist = LearningPersistence::with_config(config.clone());
        persist.initialize().unwrap();

        // Save initial state
        let mut thresholds = persist.thresholds().clone();
        thresholds.phi_threshold = 0.4;
        persist.update_thresholds(thresholds);
        persist.save().unwrap();

        // Modify and save again
        let mut thresholds = persist.thresholds().clone();
        thresholds.phi_threshold = 0.6;
        persist.update_thresholds(thresholds);
        persist.save().unwrap();

        // Rollback to version 1
        persist.rollback(1).unwrap();
        assert!((persist.thresholds().phi_threshold - 0.4).abs() < 0.001);
    }
}
