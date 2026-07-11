// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Loop State Persistence
//!
//! This module implements the "Epistemic Save File" - persistent storage for
//! calibration state that survives process restarts.
//!
//! ## The Problem
//!
//! Without persistence, the MAGI Loop has "anterograde amnesia". It learns
//! perfect calibration during a session (Brier scores drop from 0.81 to 0.21),
//! but the moment the process exits, that hard-earned wisdom is deleted.
//!
//! ## The Solution
//!
//! We persist:
//! - Per-domain calibration history (Brier scores, ECE, confidence adjustments)
//! - Constraint gate configuration (learned thresholds)
//! - Recent causal attributions (for pattern recognition)
//! - Loop state counters (to track cumulative experience)
//!
//! ## File Format
//!
//! State is saved to `~/.symthaea/magi_state.json` (configurable).
//! Uses JSON for human readability and debugging.
//!
//! ## Lifecycle
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    PERSISTENCE LIFECYCLE                        │
//! │                                                                 │
//! │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
//! │  │  COLD START │      │  WARM START │      │   RUNTIME   │    │
//! │  │ (No state)  │      │(State found)│      │  (Active)   │    │
//! │  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
//! │         │                    │                    │            │
//! │         ▼                    ▼                    ▼            │
//! │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
//! │  │  Initialize │      │    LOAD     │      │  AUTOSAVE   │    │
//! │  │   defaults  │      │   state.json│      │  every N    │    │
//! │  └──────┬──────┘      └──────┬──────┘      │  resolutions│    │
//! │         │                    │             └──────┬──────┘    │
//! │         └────────────────────┴───────────────────┘            │
//! │                              │                                 │
//! │                              ▼                                 │
//! │                    ┌─────────────────┐                        │
//! │                    │  SAVE ON EXIT   │                        │
//! │                    │ (graceful only) │                        │
//! │                    └─────────────────┘                        │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::constraint_gate::ConstraintGateConfig;
use super::magi_integration::{
    CalibrationQuality, CapabilityDomain, CausalAttribution, MagiLoopState,
};
use super::world_prediction::PredictionDomain;

// ═══════════════════════════════════════════════════════════════════════════════
// VERSION & CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Current state file version
pub const STATE_VERSION: &str = "1.0.0";

/// Default state file path
pub const DEFAULT_STATE_PATH: &str = ".symthaea/magi_state.json";

/// Configuration for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Path to state file (relative to home directory or absolute)
    pub state_path: PathBuf,

    /// Auto-save after this many resolutions (0 = disabled)
    pub autosave_interval: usize,

    /// Create backup before overwriting
    pub create_backups: bool,

    /// Maximum backups to keep
    pub max_backups: usize,

    /// Enable persistence (can be disabled for testing)
    pub enabled: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            state_path: PathBuf::from(DEFAULT_STATE_PATH),
            autosave_interval: 10,
            create_backups: true,
            max_backups: 5,
            enabled: true,
        }
    }
}

impl PersistenceConfig {
    /// Get the full path to the state file
    pub fn full_path(&self) -> io::Result<PathBuf> {
        if self.state_path.is_absolute() {
            Ok(self.state_path.clone())
        } else {
            let home = dirs::home_dir().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "Could not determine home directory",
                )
            })?;
            Ok(home.join(&self.state_path))
        }
    }

    /// Create state directory if it doesn't exist
    pub fn ensure_directory(&self) -> io::Result<()> {
        let full_path = self.full_path()?;
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTED DOMAIN CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Serializable representation of domain calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedDomainCalibration {
    /// Domain being tracked
    pub domain: PredictionDomain,

    /// Rolling Brier score
    pub rolling_brier: f64,

    /// Lifetime Brier score
    pub lifetime_brier: f64,

    /// Expected Calibration Error
    pub ece: f64,

    /// Total predictions made
    pub prediction_count: usize,

    /// Correct predictions
    pub correct_count: usize,

    /// Cumulative Brier sum (for reconstruction)
    pub brier_sum: f64,

    /// Learned confidence adjustment factor
    pub confidence_adjustment: f64,

    /// Is this domain overconfident?
    pub is_overconfident: bool,

    /// Recent prediction outcomes (limited history for rolling calculations)
    pub recent_outcomes: Vec<PersistedPredictionOutcome>,
}

/// Serializable prediction outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPredictionOutcome {
    pub confidence: f64,
    pub was_correct: bool,
    pub brier_component: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTED CAUSAL ATTRIBUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Serializable causal attribution (without Instant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedCausalAttribution {
    /// Prediction ID that was wrong
    pub prediction_id: String,

    /// Failure mode description
    pub failure_mode: String,

    /// Missing information
    pub missing_information: Vec<String>,

    /// Responsible capability domains
    pub responsible_domains: Vec<CapabilityDomain>,

    /// Prediction about recurrence
    pub recurrence_prediction: Option<String>,

    /// Confidence in this attribution
    pub confidence: f64,

    /// Unix timestamp when created
    pub created_at_unix: u64,
}

impl From<&CausalAttribution> for PersistedCausalAttribution {
    /// Convert from a reference - requires cloning all fields.
    /// Use `From<CausalAttribution>` when you have ownership to avoid clones.
    fn from(attr: &CausalAttribution) -> Self {
        Self {
            prediction_id: attr.prediction_id.clone(),
            failure_mode: attr.failure_mode.clone(),
            missing_information: attr.missing_information.clone(),
            responsible_domains: attr.responsible_domains.clone(),
            recurrence_prediction: attr.recurrence_prediction.clone(),
            confidence: attr.confidence,
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

impl From<CausalAttribution> for PersistedCausalAttribution {
    /// Convert from an owned value - moves all fields without cloning.
    fn from(attr: CausalAttribution) -> Self {
        Self {
            prediction_id: attr.prediction_id,
            failure_mode: attr.failure_mode,
            missing_information: attr.missing_information,
            responsible_domains: attr.responsible_domains,
            recurrence_prediction: attr.recurrence_prediction,
            confidence: attr.confidence,
            created_at_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAGI STATE SNAPSHOT
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete MAGI state snapshot for persistence
///
/// This is the "Epistemic Save File" - it captures everything needed to
/// restore the system's learned calibration and wisdom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagiStateSnapshot {
    /// State file version (for migration)
    pub version: String,

    /// Unix timestamp when saved
    pub saved_at_unix: u64,

    /// Human-readable save timestamp
    pub saved_at_iso: String,

    /// Session count (increments each load)
    pub session_count: u64,

    /// Total lifetime loop iterations across all sessions
    pub lifetime_iterations: usize,

    /// Per-domain calibration data
    pub calibration: HashMap<PredictionDomain, PersistedDomainCalibration>,

    /// Global calibration statistics
    pub global_stats: GlobalCalibrationStats,

    /// Constraint gate configuration (potentially learned thresholds)
    pub gate_config: ConstraintGateConfig,

    /// Recent causal attributions (for pattern learning)
    pub attribution_history: Vec<PersistedCausalAttribution>,

    /// Loop state counters
    pub loop_state: PersistedLoopState,

    /// Capability estimates per domain
    pub capability_estimates: HashMap<CapabilityDomain, f64>,

    /// Custom metadata (for extensions)
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Global calibration statistics for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCalibrationStats {
    pub rolling_brier: f64,
    pub lifetime_brier: f64,
    pub ece: f64,
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub brier_sum: f64,
    pub confidence_adjustment: f64,
    pub is_well_calibrated: bool,
}

/// Persisted loop state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedLoopState {
    pub loop_iterations: usize,
    pub predictions_made: usize,
    pub predictions_resolved: usize,
    pub attributions_generated: usize,
    pub calibration_quality: CalibrationQuality,
}

impl From<&MagiLoopState> for PersistedLoopState {
    fn from(state: &MagiLoopState) -> Self {
        Self {
            loop_iterations: state.loop_iterations,
            predictions_made: state.predictions_made,
            predictions_resolved: state.predictions_resolved,
            attributions_generated: state.attributions_generated,
            calibration_quality: state.calibration_quality,
        }
    }
}

impl MagiStateSnapshot {
    /// Create a new empty snapshot
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();

        Self {
            version: STATE_VERSION.to_string(),
            saved_at_unix: now.as_secs(),
            saved_at_iso: chrono_iso_now(),
            session_count: 0,
            lifetime_iterations: 0,
            calibration: HashMap::new(),
            global_stats: GlobalCalibrationStats {
                rolling_brier: 0.0,
                lifetime_brier: 0.0,
                ece: 0.0,
                total_predictions: 0,
                correct_predictions: 0,
                brier_sum: 0.0,
                confidence_adjustment: 1.0,
                is_well_calibrated: false,
            },
            gate_config: ConstraintGateConfig::default(),
            attribution_history: Vec::new(),
            loop_state: PersistedLoopState {
                loop_iterations: 0,
                predictions_made: 0,
                predictions_resolved: 0,
                attributions_generated: 0,
                calibration_quality: CalibrationQuality::Insufficient,
            },
            capability_estimates: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if this is a cold start (no prior state)
    pub fn is_cold_start(&self) -> bool {
        self.session_count == 0 && self.lifetime_iterations == 0
    }

    /// Get age of this snapshot
    pub fn age(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Duration::from_secs(now.saturating_sub(self.saved_at_unix))
    }

    /// Increment session count (called on load)
    pub fn bump_session(&mut self) {
        self.session_count += 1;
    }

    /// Summary for logging
    pub fn summary(&self) -> String {
        format!(
            "MagiState v{}: {} sessions, {} lifetime iterations, {} predictions, Brier={:.4}, ECE={:.4}",
            self.version,
            self.session_count,
            self.lifetime_iterations,
            self.global_stats.total_predictions,
            self.global_stats.lifetime_brier,
            self.global_stats.ece
        )
    }
}

impl Default for MagiStateSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PERSISTENCE MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Manager for MAGI state persistence
pub struct PersistenceManager {
    /// Configuration
    config: PersistenceConfig,

    /// Current snapshot (in memory)
    current: MagiStateSnapshot,

    /// Counter since last save
    unsaved_resolutions: usize,

    /// Whether state was loaded from disk
    loaded_from_disk: bool,

    /// Any load errors encountered
    load_error: Option<String>,
}

impl PersistenceManager {
    /// Create a new persistence manager
    pub fn new(config: PersistenceConfig) -> Self {
        Self {
            config,
            current: MagiStateSnapshot::new(),
            unsaved_resolutions: 0,
            loaded_from_disk: false,
            load_error: None,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(PersistenceConfig::default())
    }

    /// Initialize by loading from disk if available
    pub fn initialize(&mut self) -> io::Result<StartupMode> {
        if !self.config.enabled {
            return Ok(StartupMode::Disabled);
        }

        match self.load() {
            Ok(snapshot) => {
                self.current = snapshot;
                self.current.bump_session();
                self.loaded_from_disk = true;
                Ok(StartupMode::WarmStart {
                    session: self.current.session_count,
                    lifetime_iterations: self.current.lifetime_iterations,
                    last_saved: self.current.saved_at_iso.clone(),
                })
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                self.current = MagiStateSnapshot::new();
                self.current.bump_session();
                Ok(StartupMode::ColdStart)
            }
            Err(e) => {
                self.load_error = Some(e.to_string());
                self.current = MagiStateSnapshot::new();
                self.current.bump_session();
                Ok(StartupMode::RecoveryStart {
                    error: e.to_string(),
                })
            }
        }
    }

    /// Load state from disk
    fn load(&self) -> io::Result<MagiStateSnapshot> {
        let path = self.config.full_path()?;
        let file = fs::File::open(&path)?;
        let reader = BufReader::new(file);
        let snapshot: MagiStateSnapshot = serde_json::from_reader(reader)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(snapshot)
    }

    /// Save current state to disk
    pub fn save(&mut self) -> io::Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        self.config.ensure_directory()?;
        let path = self.config.full_path()?;

        // Create backup if configured
        if self.config.create_backups && path.exists() {
            self.create_backup(&path)?;
        }

        // Update timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        self.current.saved_at_unix = now.as_secs();
        self.current.saved_at_iso = chrono_iso_now();

        // Write atomically (write to temp, then rename)
        let temp_path = path.with_extension("json.tmp");
        {
            let file = fs::File::create(&temp_path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &self.current).map_err(io::Error::other)?;
        }

        fs::rename(&temp_path, &path)?;
        self.unsaved_resolutions = 0;

        Ok(())
    }

    /// Create a backup of the current state file
    fn create_backup(&self, path: &Path) -> io::Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let backup_name = format!(
            "{}.{}.bak",
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("magi_state"),
            timestamp
        );
        let backup_path = path.with_file_name(backup_name);

        fs::copy(path, &backup_path)?;

        // Clean up old backups
        self.cleanup_old_backups(path)?;

        Ok(())
    }

    /// Remove old backup files beyond max_backups
    fn cleanup_old_backups(&self, path: &Path) -> io::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No parent directory"))?;

        let base_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("magi_state");

        let mut backups: Vec<_> = fs::read_dir(parent)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .map(|s| s.starts_with(base_name) && s.ends_with(".bak"))
                    .unwrap_or(false)
            })
            .collect();

        if backups.len() <= self.config.max_backups {
            return Ok(());
        }

        // Sort by modification time (oldest first)
        backups.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH)
        });

        // Remove oldest backups
        let to_remove = backups.len() - self.config.max_backups;
        for entry in backups.into_iter().take(to_remove) {
            let _ = fs::remove_file(entry.path());
        }

        Ok(())
    }

    /// Record a resolution (triggers autosave if interval reached)
    pub fn on_resolution(&mut self) {
        self.unsaved_resolutions += 1;

        if self.config.autosave_interval > 0
            && self.unsaved_resolutions >= self.config.autosave_interval
        {
            // Autosave (ignore errors for now - could log)
            let _ = self.save();
        }
    }

    /// Get current snapshot (for reading)
    pub fn current(&self) -> &MagiStateSnapshot {
        &self.current
    }

    /// Get mutable snapshot (for updating)
    pub fn current_mut(&mut self) -> &mut MagiStateSnapshot {
        &mut self.current
    }

    /// Update from WorldGroundedSelfModel state
    pub fn update_from_model(&mut self, model: &super::magi_integration::WorldGroundedSelfModel) {
        // Update loop state
        let loop_state = model.loop_state();
        self.current.loop_state = PersistedLoopState::from(loop_state);
        self.current.lifetime_iterations = loop_state.loop_iterations;

        // Update global + per-domain calibration stats
        self.update_from_tracker(model.calibration());

        // Update attributions (keep last N)
        let attrs: Vec<PersistedCausalAttribution> = model
            .recent_attributions()
            .iter()
            .rev()
            .take(50)
            .map(PersistedCausalAttribution::from)
            .collect();
        self.current.attribution_history = attrs;

        // Update gate config
        self.current.gate_config = model.gate().config().clone();
    }

    /// Update the snapshot's calibration fields directly from a
    /// `BrierScoreTracker`.
    ///
    /// This is the tracker-level counterpart of [`Self::update_from_model`]:
    /// it lets components that own a bare `BrierScoreTracker` (e.g. the
    /// `Symthaea` facade's Phase 4.5/7.5 calibration) persist their state
    /// without wrapping it in a full `WorldGroundedSelfModel`. The restored
    /// tracker comes back via [`super::calibration::BrierScoreTracker::from_persisted`].
    pub fn update_from_tracker(&mut self, tracker: &super::calibration::BrierScoreTracker) {
        let cal_summary = tracker.calibration_summary();
        self.current.global_stats = GlobalCalibrationStats {
            rolling_brier: tracker.rolling_brier_score(),
            lifetime_brier: cal_summary.global_brier,
            ece: cal_summary.global_ece,
            total_predictions: cal_summary.total_predictions,
            correct_predictions: (cal_summary.global_accuracy
                * cal_summary.total_predictions as f64) as usize,
            brier_sum: cal_summary.global_brier * cal_summary.total_predictions as f64,
            confidence_adjustment: 1.0, // Global doesn't have single adjustment
            is_well_calibrated: cal_summary.is_well_calibrated,
        };

        for (domain, stats) in cal_summary.domain_stats {
            self.current.calibration.insert(
                domain,
                PersistedDomainCalibration {
                    domain,
                    rolling_brier: stats.brier_score, // Using lifetime as proxy
                    lifetime_brier: stats.brier_score,
                    ece: stats.ece,
                    prediction_count: stats.prediction_count,
                    correct_count: (stats.accuracy * stats.prediction_count as f64) as usize,
                    brier_sum: stats.brier_score * stats.prediction_count as f64,
                    confidence_adjustment: stats.confidence_adjustment,
                    is_overconfident: stats.is_overconfident,
                    recent_outcomes: Vec::new(), // Don't persist all outcomes
                },
            );
        }
    }

    /// Check if we loaded from disk
    pub fn was_warm_start(&self) -> bool {
        self.loaded_from_disk
    }

    /// Get any load error
    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    /// Get pending (unsaved) resolution count
    pub fn unsaved_count(&self) -> usize {
        self.unsaved_resolutions
    }

    /// Force save (ignoring autosave interval)
    pub fn force_save(&mut self) -> io::Result<()> {
        self.save()
    }

    /// Get configuration
    pub fn config(&self) -> &PersistenceConfig {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STARTUP MODE
// ═══════════════════════════════════════════════════════════════════════════════

/// How the system started
#[derive(Debug, Clone)]
pub enum StartupMode {
    /// No prior state found - starting fresh
    ColdStart,

    /// Successfully loaded prior state
    WarmStart {
        session: u64,
        lifetime_iterations: usize,
        last_saved: String,
    },

    /// Found state but failed to load - starting fresh
    RecoveryStart { error: String },

    /// Persistence disabled
    Disabled,
}

impl StartupMode {
    /// Human-readable description
    pub fn description(&self) -> String {
        match self {
            Self::ColdStart => "Cold start (no prior state)".to_string(),
            Self::WarmStart {
                session,
                lifetime_iterations,
                last_saved,
            } => {
                format!(
                    "Warm start (session #{}, {} lifetime iterations, last saved {})",
                    session, lifetime_iterations, last_saved
                )
            }
            Self::RecoveryStart { error } => {
                format!("Recovery start (failed to load: {})", error)
            }
            Self::Disabled => "Persistence disabled".to_string(),
        }
    }

    /// Is this a warm start with prior knowledge?
    pub fn has_prior_knowledge(&self) -> bool {
        matches!(self, Self::WarmStart { .. })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DROP IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

impl Drop for PersistenceManager {
    /// Best-effort save on drop
    ///
    /// This ensures that even if the program exits unexpectedly (but not via
    /// panic or signal), the learned calibration is preserved.
    fn drop(&mut self) {
        if self.config.enabled && self.unsaved_resolutions > 0 {
            // Best-effort save - ignore errors during drop
            let _ = self.save();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAGI PERSISTENT MODEL (High-Level Integration)
// ═══════════════════════════════════════════════════════════════════════════════

/// A WorldGroundedSelfModel with automatic persistence
///
/// This is the recommended way to use the MAGI Loop with persistence.
/// It automatically:
/// - Loads prior state on creation (warm start if available)
/// - Saves state after each resolution (via autosave)
/// - Saves state on drop (best-effort)
///
/// # Example
///
/// ```ignore
/// // Create a persistent model - automatically loads prior state
/// let mut magi = MagiPersistentModel::new()?;
///
/// println!("Startup: {}", magi.startup_mode().description());
///
/// // Use it like a normal WorldGroundedSelfModel
/// let prediction = magi.model_mut().predict(
///     "File will be created",
///     OutcomeCategory::Success,
///     0.85,
///     action,
/// );
///
/// // ... resolve prediction ...
///
/// // Sync and save (also happens automatically on drop)
/// magi.sync_and_save()?;
/// ```
pub struct MagiPersistentModel {
    /// The underlying world-grounded self model
    model: super::magi_integration::WorldGroundedSelfModel,

    /// Persistence manager
    persistence: PersistenceManager,

    /// How we started (for diagnostics)
    startup_mode: StartupMode,
}

impl MagiPersistentModel {
    /// Create a new persistent model with default configuration
    ///
    /// Automatically loads prior state if available (warm start).
    pub fn new() -> io::Result<Self> {
        Self::with_config(PersistenceConfig::default())
    }

    /// Create with custom persistence configuration
    pub fn with_config(config: PersistenceConfig) -> io::Result<Self> {
        let mut persistence = PersistenceManager::new(config);
        let startup_mode = persistence.initialize()?;

        let model = match &startup_mode {
            StartupMode::WarmStart { .. } => {
                super::magi_integration::WorldGroundedSelfModel::from_snapshot(
                    persistence.current(),
                )
            }
            _ => super::magi_integration::WorldGroundedSelfModel::with_defaults(),
        };

        Ok(Self {
            model,
            persistence,
            startup_mode,
        })
    }

    /// Create with custom model configuration and persistence configuration
    pub fn with_configs(
        model_config: super::magi_integration::WorldGroundedConfig,
        persistence_config: PersistenceConfig,
    ) -> io::Result<Self> {
        let mut persistence = PersistenceManager::new(persistence_config);
        let startup_mode = persistence.initialize()?;

        let model = match &startup_mode {
            StartupMode::WarmStart { .. } => {
                // Merge snapshot with custom config where appropriate
                let restored = super::magi_integration::WorldGroundedSelfModel::from_snapshot(
                    persistence.current(),
                );
                // The snapshot takes precedence for learned state,
                // but we could merge other config here if needed
                restored
            }
            _ => super::magi_integration::WorldGroundedSelfModel::new(model_config),
        };

        Ok(Self {
            model,
            persistence,
            startup_mode,
        })
    }

    /// Get the startup mode (for diagnostics)
    pub fn startup_mode(&self) -> &StartupMode {
        &self.startup_mode
    }

    /// Get immutable reference to the model
    pub fn model(&self) -> &super::magi_integration::WorldGroundedSelfModel {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut super::magi_integration::WorldGroundedSelfModel {
        &mut self.model
    }

    /// Record a resolution and trigger autosave if needed
    ///
    /// Call this after resolving a prediction to sync persistence.
    pub fn on_resolution(&mut self) {
        self.persistence.update_from_model(&self.model);
        self.persistence.on_resolution();
    }

    /// Sync model state to persistence and force save
    pub fn sync_and_save(&mut self) -> io::Result<()> {
        self.persistence.update_from_model(&self.model);
        self.persistence.force_save()
    }

    /// Get persistence manager (for advanced usage)
    pub fn persistence(&self) -> &PersistenceManager {
        &self.persistence
    }

    /// Get mutable persistence manager (for advanced usage)
    pub fn persistence_mut(&mut self) -> &mut PersistenceManager {
        &mut self.persistence
    }

    /// Get lifetime summary
    pub fn lifetime_summary(&self) -> String {
        format!(
            "{}\nSession: #{}, Unsaved: {}",
            self.model.lifetime_summary(),
            self.persistence.current().session_count,
            self.persistence.unsaved_count()
        )
    }

    /// Was this a warm start with prior knowledge?
    pub fn has_prior_knowledge(&self) -> bool {
        self.startup_mode.has_prior_knowledge()
    }
}

impl Drop for MagiPersistentModel {
    fn drop(&mut self) {
        // Sync current state before the persistence manager's drop saves it
        self.persistence.update_from_model(&self.model);
        // PersistenceManager's drop will handle the actual save
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Get current time as ISO 8601 string (without external chrono crate)
fn chrono_iso_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    // Simple ISO format: seconds since epoch
    // For proper ISO 8601, would need chrono, but keeping deps minimal
    format!("{}Z", now.as_secs())
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_config(temp_dir: &TempDir) -> PersistenceConfig {
        PersistenceConfig {
            state_path: temp_dir.path().join("test_magi_state.json"),
            autosave_interval: 5,
            create_backups: true,
            max_backups: 3,
            enabled: true,
        }
    }

    #[test]
    fn test_cold_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let mut manager = PersistenceManager::new(config);
        let mode = manager.initialize().unwrap();

        assert!(matches!(mode, StartupMode::ColdStart));
        assert!(!manager.was_warm_start());
        assert_eq!(manager.current().session_count, 1);
    }

    #[test]
    fn test_save_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // First session: create and save
        {
            let mut manager = PersistenceManager::new(config.clone());
            manager.initialize().unwrap();

            // Modify state
            manager.current_mut().lifetime_iterations = 42;
            manager.current_mut().global_stats.total_predictions = 100;
            manager.current_mut().global_stats.lifetime_brier = 0.21;

            manager.save().unwrap();
        }

        // Second session: load
        {
            let mut manager = PersistenceManager::new(config);
            let mode = manager.initialize().unwrap();

            assert!(matches!(mode, StartupMode::WarmStart { .. }));
            assert!(manager.was_warm_start());
            assert_eq!(manager.current().session_count, 2);
            assert_eq!(manager.current().lifetime_iterations, 42);
            assert_eq!(manager.current().global_stats.total_predictions, 100);
            assert!((manager.current().global_stats.lifetime_brier - 0.21).abs() < 0.001);
        }
    }

    #[test]
    fn test_autosave() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = test_config(&temp_dir);
        config.autosave_interval = 3;

        let mut manager = PersistenceManager::new(config.clone());
        manager.initialize().unwrap();

        // Record resolutions
        manager.on_resolution();
        assert_eq!(manager.unsaved_count(), 1);
        manager.on_resolution();
        assert_eq!(manager.unsaved_count(), 2);
        manager.on_resolution(); // Should trigger autosave
        assert_eq!(manager.unsaved_count(), 0);

        // Verify file exists
        assert!(config.full_path().unwrap().exists());
    }

    #[test]
    fn test_backup_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let mut manager = PersistenceManager::new(config.clone());
        manager.initialize().unwrap();

        // Save multiple times
        for i in 0..5 {
            manager.current_mut().lifetime_iterations = i;
            manager.save().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
        }

        // Check backups exist (max 3)
        let backup_count = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.file_name().to_str().map(|s| s.ends_with(".bak")))
                    .unwrap_or(false)
            })
            .count();

        assert!(backup_count <= 3);
    }

    #[test]
    fn test_domain_calibration_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let mut manager = PersistenceManager::new(config.clone());
        manager.initialize().unwrap();

        // Add domain calibration
        manager.current_mut().calibration.insert(
            PredictionDomain::CodeExecution,
            PersistedDomainCalibration {
                domain: PredictionDomain::CodeExecution,
                rolling_brier: 0.15,
                lifetime_brier: 0.18,
                ece: 0.05,
                prediction_count: 50,
                correct_count: 42,
                brier_sum: 9.0,
                confidence_adjustment: 0.85,
                is_overconfident: true,
                recent_outcomes: vec![PersistedPredictionOutcome {
                    confidence: 0.9,
                    was_correct: true,
                    brier_component: 0.01,
                }],
            },
        );

        manager.save().unwrap();

        // Reload
        let mut manager2 = PersistenceManager::new(config);
        manager2.initialize().unwrap();

        let cal = manager2
            .current()
            .calibration
            .get(&PredictionDomain::CodeExecution)
            .unwrap();
        assert_eq!(cal.prediction_count, 50);
        assert!((cal.confidence_adjustment - 0.85).abs() < 0.001);
        assert!(cal.is_overconfident);
    }

    #[test]
    fn test_disabled_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = test_config(&temp_dir);
        config.enabled = false;

        let mut manager = PersistenceManager::new(config.clone());
        let mode = manager.initialize().unwrap();

        assert!(matches!(mode, StartupMode::Disabled));

        // Save should be no-op
        manager.save().unwrap();
        assert!(!config.full_path().unwrap().exists());
    }

    #[test]
    fn test_snapshot_summary() {
        let mut snapshot = MagiStateSnapshot::new();
        snapshot.session_count = 5;
        snapshot.lifetime_iterations = 100;
        snapshot.global_stats.total_predictions = 500;
        snapshot.global_stats.lifetime_brier = 0.18;
        snapshot.global_stats.ece = 0.07;

        let summary = snapshot.summary();
        assert!(summary.contains("5 sessions"));
        assert!(summary.contains("100 lifetime iterations"));
        assert!(summary.contains("500 predictions"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGRATION TESTS: Warm Start with MagiPersistentModel
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_magi_persistent_model_cold_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let magi = MagiPersistentModel::with_config(config).unwrap();

        // Verify cold start
        assert!(matches!(magi.startup_mode(), StartupMode::ColdStart));
        assert!(!magi.has_prior_knowledge());

        // Model should be fresh (no prior iterations)
        assert!(!magi.model().is_warm_start());
    }

    #[test]
    fn test_magi_persistent_model_warm_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Session 1: Create model and record some work
        {
            let mut magi = MagiPersistentModel::with_config(config.clone()).unwrap();
            assert!(matches!(magi.startup_mode(), StartupMode::ColdStart));

            // Record that we've done work (directly via persistence snapshot)
            magi.persistence_mut().current_mut().lifetime_iterations = 10;
            magi.persistence_mut()
                .current_mut()
                .global_stats
                .total_predictions = 25;
            magi.persistence_mut()
                .current_mut()
                .global_stats
                .lifetime_brier = 0.22;
            magi.persistence_mut().current_mut().global_stats.ece = 0.08;

            // Also set loop state so warm start detection works
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .loop_iterations = 10;
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .predictions_made = 25;

            // Force save directly (skip update_from_model since we set values manually)
            magi.persistence_mut().force_save().unwrap();
        }

        // Session 2: Warm start should restore state
        {
            let magi = MagiPersistentModel::with_config(config).unwrap();

            // Verify warm start
            assert!(magi.has_prior_knowledge());
            if let StartupMode::WarmStart {
                session,
                lifetime_iterations,
                ..
            } = magi.startup_mode()
            {
                assert_eq!(*session, 2);
                assert_eq!(*lifetime_iterations, 10);
            } else {
                panic!("Expected warm start");
            }

            // Verify state was restored
            let snapshot = magi.persistence().current();
            assert_eq!(snapshot.session_count, 2);
            assert_eq!(snapshot.lifetime_iterations, 10);
            assert_eq!(snapshot.global_stats.total_predictions, 25);
            assert!((snapshot.global_stats.lifetime_brier - 0.22).abs() < 0.001);
            assert!((snapshot.global_stats.ece - 0.08).abs() < 0.001);
        }
    }

    #[test]
    fn test_magi_persistent_model_calibration_restore() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Session 1: Build up calibration
        {
            let mut magi = MagiPersistentModel::with_config(config.clone()).unwrap();

            // Add domain-specific calibration (using SystemState domain)
            magi.persistence_mut().current_mut().calibration.insert(
                PredictionDomain::SystemState,
                PersistedDomainCalibration {
                    domain: PredictionDomain::SystemState,
                    rolling_brier: 0.12,
                    lifetime_brier: 0.15,
                    ece: 0.04,
                    prediction_count: 100,
                    correct_count: 88,
                    brier_sum: 15.0,
                    confidence_adjustment: 0.92,
                    is_overconfident: false,
                    recent_outcomes: vec![
                        PersistedPredictionOutcome {
                            confidence: 0.85,
                            was_correct: true,
                            brier_component: 0.0225,
                        },
                        PersistedPredictionOutcome {
                            confidence: 0.70,
                            was_correct: true,
                            brier_component: 0.09,
                        },
                    ],
                },
            );

            // Add causal attribution
            magi.persistence_mut()
                .current_mut()
                .attribution_history
                .push(PersistedCausalAttribution {
                    prediction_id: "test-prediction-001".to_string(),
                    failure_mode: "External system timeout".to_string(),
                    missing_information: vec!["network latency".to_string()],
                    responsible_domains: vec![CapabilityDomain::Memory],
                    recurrence_prediction: Some("May recur under high load".to_string()),
                    confidence: 0.85,
                    created_at_unix: 1700000000,
                });

            // Set loop state
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .loop_iterations = 50;
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .predictions_made = 200;
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .predictions_resolved = 195;

            // Force save directly (skip update_from_model since we set values manually)
            magi.persistence_mut().force_save().unwrap();
        }

        // Session 2: Verify calibration was restored
        {
            let magi = MagiPersistentModel::with_config(config).unwrap();
            assert!(magi.has_prior_knowledge());

            let snapshot = magi.persistence().current();

            // Check domain calibration
            let sys_cal = snapshot
                .calibration
                .get(&PredictionDomain::SystemState)
                .unwrap();
            assert_eq!(sys_cal.prediction_count, 100);
            assert_eq!(sys_cal.correct_count, 88);
            assert!((sys_cal.confidence_adjustment - 0.92).abs() < 0.001);
            assert!(!sys_cal.is_overconfident);
            assert_eq!(sys_cal.recent_outcomes.len(), 2);

            // Check causal attribution
            assert_eq!(snapshot.attribution_history.len(), 1);
            assert_eq!(
                snapshot.attribution_history[0].prediction_id,
                "test-prediction-001"
            );
            assert!((snapshot.attribution_history[0].confidence - 0.85).abs() < 0.001);

            // Check loop state
            assert_eq!(snapshot.loop_state.loop_iterations, 50);
            assert_eq!(snapshot.loop_state.predictions_made, 200);
            assert_eq!(snapshot.loop_state.predictions_resolved, 195);
        }
    }

    #[test]
    fn test_magi_persistent_model_lifetime_summary() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        let mut magi = MagiPersistentModel::with_config(config).unwrap();
        magi.persistence_mut().current_mut().lifetime_iterations = 42;

        let summary = magi.lifetime_summary();
        assert!(summary.contains("Session: #1"));
        assert!(summary.contains("Unsaved:"));
    }

    #[test]
    fn test_magi_persistent_model_on_resolution() {
        let temp_dir = TempDir::new().unwrap();
        let mut config = test_config(&temp_dir);
        config.autosave_interval = 2;

        let mut magi = MagiPersistentModel::with_config(config.clone()).unwrap();

        // First resolution
        magi.on_resolution();
        assert_eq!(magi.persistence().unsaved_count(), 1);

        // Second resolution should trigger autosave
        magi.on_resolution();
        assert_eq!(magi.persistence().unsaved_count(), 0);

        // Verify file exists
        assert!(config.full_path().unwrap().exists());
    }

    #[test]
    fn test_magi_persistent_model_multiple_sessions() {
        let temp_dir = TempDir::new().unwrap();
        let config = test_config(&temp_dir);

        // Run 5 sessions
        for i in 1..=5 {
            let mut magi = MagiPersistentModel::with_config(config.clone()).unwrap();

            if i == 1 {
                assert!(matches!(magi.startup_mode(), StartupMode::ColdStart));
            } else {
                assert!(magi.has_prior_knowledge());
                if let StartupMode::WarmStart { session, .. } = magi.startup_mode() {
                    assert_eq!(*session, i as u64);
                }
            }

            // Increment lifetime iterations
            let current_iterations = magi.persistence().current().lifetime_iterations;
            magi.persistence_mut().current_mut().lifetime_iterations = current_iterations + 10;
            // Also update loop_state for warm start detection
            magi.persistence_mut()
                .current_mut()
                .loop_state
                .loop_iterations = current_iterations + 10;
            // Force save directly (skip update_from_model since we set values manually)
            magi.persistence_mut().force_save().unwrap();
        }

        // Final verification
        let magi = MagiPersistentModel::with_config(config).unwrap();
        let snapshot = magi.persistence().current();
        assert_eq!(snapshot.session_count, 6);
        assert_eq!(snapshot.lifetime_iterations, 50); // 5 sessions * 10 iterations
    }
}
