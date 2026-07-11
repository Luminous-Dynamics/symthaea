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

/// Per-outcome threshold adjustment step size.
const THRESHOLD_STEP: f64 = 0.01;
/// Lower clamp for adaptive thresholds.
const THRESHOLD_MIN: f64 = 0.05;
/// Upper clamp for adaptive thresholds.
const THRESHOLD_MAX: f64 = 0.95;
/// Cap on retained threshold-adjustment history entries.
const MAX_ADJUSTMENT_HISTORY: usize = 256;

impl AdaptiveThresholds {
    /// Record a verified interaction outcome and adapt thresholds.
    ///
    /// Simple threshold servo: a *failure* that passed a threshold pushes
    /// that threshold up (be stricter); a *success* that fell below a
    /// threshold pulls it down (we were too strict). Steps are small
    /// (`THRESHOLD_STEP`) and clamped to `[THRESHOLD_MIN, THRESHOLD_MAX]`.
    /// Every actual adjustment is appended to `adjustment_history` as
    /// `(unix_millis, old, new)`, capped at `MAX_ADJUSTMENT_HISTORY`.
    pub fn record_outcome(&mut self, phi: f64, confidence: f64, success: bool) {
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Confidence threshold servo
        let old_conf = self.confidence_threshold;
        if !success && confidence >= self.confidence_threshold {
            self.confidence_threshold =
                (self.confidence_threshold + THRESHOLD_STEP).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        } else if success && confidence < self.confidence_threshold {
            self.confidence_threshold =
                (self.confidence_threshold - THRESHOLD_STEP).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        }
        if (self.confidence_threshold - old_conf).abs() > f64::EPSILON {
            self.adjustment_history
                .push((now, old_conf, self.confidence_threshold));
        }

        // Phi threshold servo (same shape)
        let old_phi = self.phi_threshold;
        if !success && phi >= self.phi_threshold {
            self.phi_threshold =
                (self.phi_threshold + THRESHOLD_STEP).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        } else if success && phi < self.phi_threshold {
            self.phi_threshold =
                (self.phi_threshold - THRESHOLD_STEP).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        }
        if (self.phi_threshold - old_phi).abs() > f64::EPSILON {
            self.adjustment_history
                .push((now, old_phi, self.phi_threshold));
        }

        // Bound history growth
        if self.adjustment_history.len() > MAX_ADJUSTMENT_HISTORY {
            let excess = self.adjustment_history.len() - MAX_ADJUSTMENT_HISTORY;
            self.adjustment_history.drain(..excess);
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

impl OutcomePatterns {
    /// Record an outcome for a quadrant, updating the running success rate
    /// incrementally. Quadrant layout matches the struct docs:
    /// 0 = low-φ/low-conf, 1 = low-φ/high-conf, 2 = high-φ/low-conf,
    /// 3 = high-φ/high-conf.
    pub fn record(&mut self, quadrant: usize, success: bool) {
        let q = quadrant.min(3);
        let n = self.quadrant_samples[q] as f64;
        let outcome = if success { 1.0 } else { 0.0 };
        self.quadrant_success_rates[q] = (self.quadrant_success_rates[q] * n + outcome) / (n + 1.0);
        self.quadrant_samples[q] = self.quadrant_samples[q].saturating_add(1);
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

    /// Record a per-interaction outcome into the adaptive learning state.
    ///
    /// This is what makes the persisted `LearningState` genuinely adaptive:
    /// the facade calls it after fidelity verification with the interaction's
    /// phi (`StructuredThought::psi`), the generation confidence, and whether
    /// the translation was verified. It updates the quadrant outcome patterns
    /// (bucketed by the thresholds *before* adjustment) and then runs the
    /// threshold servo in [`AdaptiveThresholds::record_outcome`].
    pub fn record_outcome(&mut self, phi: f64, confidence: f64, success: bool) {
        // Bucket by the current thresholds before they adapt.
        let hi_phi = phi >= self.state.thresholds.phi_threshold;
        let hi_conf = confidence >= self.state.thresholds.confidence_threshold;
        let quadrant = (hi_phi as usize) * 2 + (hi_conf as usize);
        self.state.patterns.record(quadrant, success);
        self.state
            .thresholds
            .record_outcome(phi, confidence, success);
        self.dirty = true;
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
    fn test_record_outcome_updates_counts_and_quadrants() {
        let mut persist = LearningPersistence::new();

        // high-phi/high-conf success (defaults: phi 0.5, conf 0.6) -> quadrant 3
        persist.record_outcome(0.8, 0.9, true);
        assert_eq!(persist.thresholds().successes, 1);
        assert_eq!(persist.thresholds().failures, 0);
        assert_eq!(persist.patterns().quadrant_samples[3], 1);
        assert!((persist.patterns().quadrant_success_rates[3] - 1.0).abs() < 1e-9);

        // low-phi/low-conf failure -> quadrant 0
        persist.record_outcome(0.1, 0.1, false);
        assert_eq!(persist.thresholds().failures, 1);
        assert_eq!(persist.patterns().quadrant_samples[0], 1);
        assert!(persist.patterns().quadrant_success_rates[0].abs() < 1e-9);

        assert!(persist.is_dirty());
    }

    #[test]
    fn test_record_outcome_adapts_thresholds() {
        let mut persist = LearningPersistence::new();
        let initial_conf = persist.thresholds().confidence_threshold;

        // Confident failures should raise the confidence threshold.
        for _ in 0..5 {
            persist.record_outcome(0.9, 0.95, false);
        }
        assert!(persist.thresholds().confidence_threshold > initial_conf);
        assert!(!persist.thresholds().adjustment_history.is_empty());

        // Low-confidence successes should pull it back down.
        let raised = persist.thresholds().confidence_threshold;
        for _ in 0..5 {
            persist.record_outcome(0.1, 0.1, true);
        }
        assert!(persist.thresholds().confidence_threshold < raised);
    }

    #[test]
    fn test_record_outcome_running_success_rate() {
        let mut patterns = OutcomePatterns::default();
        patterns.record(2, true);
        patterns.record(2, false);
        patterns.record(2, true);
        assert_eq!(patterns.quadrant_samples[2], 3);
        assert!((patterns.quadrant_success_rates[2] - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_outcome_persists_across_sessions() {
        let dir = tempdir().unwrap();
        let config = LearningPersistenceConfig::default().with_dir(dir.path().to_path_buf());

        let mut persist = LearningPersistence::with_config(config.clone());
        persist.initialize().unwrap();
        persist.record_outcome(0.8, 0.9, true);
        persist.record_outcome(0.8, 0.9, false);
        persist.save().unwrap();

        let mut persist2 = LearningPersistence::with_config(config);
        persist2.initialize().unwrap();
        assert_eq!(persist2.thresholds().successes, 1);
        assert_eq!(persist2.thresholds().failures, 1);
        assert_eq!(persist2.patterns().quadrant_samples[3], 2);
    }

    #[test]
    fn test_adjustment_history_bounded() {
        let mut thresholds = AdaptiveThresholds::default();
        // Alternate confident failures / unconfident successes so every
        // outcome adjusts both thresholds.
        for i in 0..600 {
            thresholds.record_outcome(0.9, 0.9, i % 2 == 0);
        }
        assert!(thresholds.adjustment_history.len() <= MAX_ADJUSTMENT_HISTORY);
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
