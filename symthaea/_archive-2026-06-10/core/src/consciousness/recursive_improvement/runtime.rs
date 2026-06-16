// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MAGI Loop Runtime - The Heartbeat of Symthaea
//!
//! This module implements the continuous cognitive cycle that transforms MAGI
//! from a CLI tool into a living agent.
//!
//! ## The 6-Step Autonomic Cycle
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────┐
//! │                    THE MAGI RUNTIME CYCLE                              │
//! │                                                                        │
//! │   ┌──────────┐                                    ┌──────────┐        │
//! │   │ 1. PULSE │──┐                          ┌────▶│6.CALIBRATE│        │
//! │   └──────────┘  │                          │     └──────────┘        │
//! │                 ▼                          │            │             │
//! │   ┌──────────────────┐    ┌──────────┐    │            ▼             │
//! │   │   2. PERCEIVE    │───▶│3. PREDICT│────┤     ┌──────────┐        │
//! │   │ (Bridge Signals) │    └──────────┘    │     │ PERSIST  │        │
//! │   └──────────────────┘          │         │     └──────────┘        │
//! │                                 ▼         │                          │
//! │                          ┌──────────┐     │                          │
//! │                          │ 4. GATE  │─────┤                          │
//! │                          └──────────┘     │                          │
//! │                                │          │                          │
//! │                                ▼          │                          │
//! │                          ┌──────────┐    │                          │
//! │                          │ 5. RESOLVE│───┘                          │
//! │                          └──────────┘                               │
//! └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::recursive_improvement::runtime::{
//!     MagiLoopRuntime, RuntimeConfig, RuntimeState,
//! };
//!
//! // Create runtime
//! let mut runtime = MagiLoopRuntime::new(RuntimeConfig::default())?;
//!
//! // Run in background
//! runtime.start().await?;
//!
//! // Get real-time snapshot for monitoring
//! let snapshot = runtime.snapshot();
//! println!("Tick: {} | State: {:?}", snapshot.tick_count, snapshot.state);
//! ```

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::interval;

use super::{
    ActiveInferenceBridge, ActiveInferenceBridgeConfig, CalibrationQuality, ConstraintGate,
    ConstraintGateConfig, DryRunReason, ExecutionMode, MagiPersistentModel, OutcomeCategory,
    PersistenceConfig, PredictionDomain, SupervisionReason, WorldPrediction,
};

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// The operational state of the MAGI runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RuntimeState {
    /// Initial state before first tick
    #[default]
    Initializing,
    /// Normal operation - making and resolving predictions
    Running,
    /// Observing only - high uncertainty, gathering data
    Observing,
    /// Paused by user or gate
    Paused,
    /// Temporarily idle (no pending work)
    Idle,
    /// Shutting down gracefully
    Stopping,
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the MAGI runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Tick interval in milliseconds
    pub tick_interval_ms: u64,

    /// Maximum predictions to queue
    pub max_pending_predictions: usize,

    /// Uncertainty threshold to enter Observing mode
    pub observation_threshold: f32,

    /// Auto-persist interval (0 = every tick)
    pub persist_interval_ticks: u64,

    /// Enable bridge signal generation
    pub enable_bridge_signals: bool,

    /// Log buffer size for monitor
    pub log_buffer_size: usize,

    /// Persistence configuration
    pub persistence_config: PersistenceConfig,

    /// Bridge configuration
    pub bridge_config: ActiveInferenceBridgeConfig,

    /// Gate configuration
    pub gate_config: ConstraintGateConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            tick_interval_ms: 1000, // 1 second default
            max_pending_predictions: 100,
            observation_threshold: 0.4,
            persist_interval_ticks: 10,
            enable_bridge_signals: true,
            log_buffer_size: 100,
            persistence_config: PersistenceConfig::default(),
            bridge_config: ActiveInferenceBridgeConfig::default(),
            gate_config: ConstraintGateConfig::default(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME SIGNALS (for monitoring)
// ═══════════════════════════════════════════════════════════════════════════════

/// Real-time signals from the runtime for monitoring
#[derive(Debug, Clone, Default)]
pub struct RuntimeSignals {
    /// Current prediction error (from Brier)
    pub prediction_error: f32,
    /// Current uncertainty (from ECE)
    pub uncertainty: f32,
    /// Current confidence
    pub confidence: f32,
    /// Current coherence (attribution quality)
    pub coherence: f32,
    /// Free energy estimate
    pub free_energy: f32,
    /// Whether the system is "surprised"
    pub surprise: f32,
}

/// Log entry for the runtime monitor
#[derive(Debug, Clone)]
pub struct RuntimeLogEntry {
    /// Timestamp (ticks since start)
    pub tick: u64,
    /// Wall clock time
    pub timestamp: Instant,
    /// Log level
    pub level: LogLevel,
    /// Message
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Success,
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME SNAPSHOT (thread-safe read)
// ═══════════════════════════════════════════════════════════════════════════════

/// A snapshot of the runtime state for monitoring
#[derive(Debug, Clone)]
pub struct RuntimeSnapshot {
    /// Current state
    pub state: RuntimeState,
    /// Total ticks since start
    pub tick_count: u64,
    /// Session number
    pub session_number: u64,
    /// Last tick duration in microseconds
    pub last_tick_us: u64,
    /// Average tick duration in microseconds
    pub avg_tick_us: u64,
    /// Current execution mode
    pub execution_mode: ExecutionMode,
    /// Gate status (open/restricted)
    pub gate_open: bool,
    /// Current signals
    pub signals: RuntimeSignals,
    /// Current Brier score
    pub brier_score: f64,
    /// Current ECE
    pub ece: f64,
    /// Predictions made this session
    pub predictions_made: u64,
    /// Predictions resolved this session
    pub predictions_resolved: u64,
    /// Pending predictions count
    pub pending_predictions: usize,
    /// Current prediction (if any)
    pub current_prediction: Option<String>,
    /// Recent log entries
    pub recent_logs: Vec<RuntimeLogEntry>,
    /// Per-domain Brier scores
    pub domain_briers: Vec<(PredictionDomain, f64)>,
    /// Calibration quality
    pub calibration_quality: CalibrationQuality,
    /// Lifetime predictions
    pub lifetime_predictions: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PENDING PREDICTION
// ═══════════════════════════════════════════════════════════════════════════════

/// A prediction waiting for resolution
#[derive(Debug, Clone)]
pub struct PendingPrediction {
    /// The prediction
    pub prediction: WorldPrediction,
    /// When it was made
    pub created_at: Instant,
    /// Resolution deadline
    pub deadline: Option<Instant>,
    /// Auto-resolve type (if any)
    pub auto_resolve: Option<AutoResolveType>,
}

/// Types of automatic resolution
#[derive(Debug, Clone)]
pub enum AutoResolveType {
    /// Check if a file exists
    FileExists(String),
    /// Check if a command succeeds
    CommandSucceeds(String),
    /// Check if a port is open
    PortOpen(String, u16),
    /// Check if a service is running
    ServiceRunning(String),
    /// Timeout (assume failure)
    Timeout(Duration),
}

// ═══════════════════════════════════════════════════════════════════════════════
// THE MAGI LOOP RUNTIME
// ═══════════════════════════════════════════════════════════════════════════════

/// The MAGI Loop Runtime - the heartbeat of Symthaea
///
/// This is the core autonomic system that runs the continuous cognitive cycle:
/// Pulse → Perceive → Predict → Gate → Resolve → Calibrate → Persist → Loop
pub struct MagiLoopRuntime {
    /// Configuration
    config: RuntimeConfig,

    /// The persistent model (MAGI core)
    model: Arc<Mutex<MagiPersistentModel>>,

    /// Active Inference bridge for signal generation
    bridge: Arc<Mutex<ActiveInferenceBridge>>,

    /// Constraint gate for safety
    _gate: Arc<Mutex<ConstraintGate>>,

    /// Current runtime state
    state: Arc<RwLock<RuntimeState>>,

    /// Tick counter
    tick_count: Arc<AtomicU64>,

    /// Running flag
    running: Arc<AtomicBool>,

    /// Pending predictions queue
    pending: Arc<Mutex<VecDeque<PendingPrediction>>>,

    /// Log buffer for monitoring
    logs: Arc<Mutex<VecDeque<RuntimeLogEntry>>>,

    /// Last tick duration
    last_tick_us: Arc<AtomicU64>,

    /// Total tick time for averaging
    total_tick_us: Arc<AtomicU64>,

    /// Current signals (cached for monitoring)
    signals: Arc<RwLock<RuntimeSignals>>,

    /// Event sender for external notifications
    event_tx: Option<mpsc::Sender<RuntimeEvent>>,
}

/// Events emitted by the runtime
#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    /// State changed
    StateChanged(RuntimeState),
    /// Prediction made
    PredictionMade(String),
    /// Prediction resolved
    PredictionResolved { success: bool, brier: f64 },
    /// Gate status changed
    GateChanged { open: bool, mode: ExecutionMode },
    /// Calibration updated
    CalibrationUpdated { brier: f64, ece: f64 },
    /// Error occurred
    Error(String),
}

impl MagiLoopRuntime {
    /// Create a new runtime with configuration
    pub fn new(config: RuntimeConfig) -> anyhow::Result<Self> {
        // Initialize the persistent model
        let model = MagiPersistentModel::with_config(config.persistence_config.clone())?;

        // Initialize bridge
        let bridge = ActiveInferenceBridge::new(config.bridge_config.clone());

        // Initialize gate
        let gate = ConstraintGate::new(config.gate_config.clone());

        Ok(Self {
            config,
            model: Arc::new(Mutex::new(model)),
            bridge: Arc::new(Mutex::new(bridge)),
            _gate: Arc::new(Mutex::new(gate)),
            state: Arc::new(RwLock::new(RuntimeState::Initializing)),
            tick_count: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            pending: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            logs: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            last_tick_us: Arc::new(AtomicU64::new(0)),
            total_tick_us: Arc::new(AtomicU64::new(0)),
            signals: Arc::new(RwLock::new(RuntimeSignals::default())),
            event_tx: None,
        })
    }

    /// Create with default configuration
    pub fn with_defaults() -> anyhow::Result<Self> {
        Self::new(RuntimeConfig::default())
    }

    /// Set event channel for notifications
    pub fn with_event_channel(mut self, tx: mpsc::Sender<RuntimeEvent>) -> Self {
        self.event_tx = Some(tx);
        self
    }

    /// Get current state
    pub fn state(&self) -> RuntimeState {
        *self.state.read()
    }

    /// Get a snapshot for monitoring (thread-safe)
    ///
    /// Lock ordering: pending → model → logs (consistent with resolve_pending
    /// which acquires pending → model → bridge, preventing ABBA deadlock).
    pub fn snapshot(&self) -> RuntimeSnapshot {
        let pending = self.pending.lock();
        let model = self.model.lock();
        let persistence = model.persistence();
        let current = persistence.current();
        let signals = self.signals.read().clone();
        let logs = self.logs.lock();

        let tick_count = self.tick_count.load(Ordering::Relaxed);
        let total_us = self.total_tick_us.load(Ordering::Relaxed);
        let avg_tick_us = if tick_count > 0 {
            total_us / tick_count
        } else {
            0
        };

        // Get domain briers
        let domain_briers: Vec<_> = current
            .calibration
            .iter()
            .map(|(d, c)| (*d, c.lifetime_brier))
            .collect();

        // Get current prediction
        let current_prediction = pending.front().map(|p| p.prediction.claim.clone());

        RuntimeSnapshot {
            state: *self.state.read(),
            tick_count,
            session_number: current.session_count,
            last_tick_us: self.last_tick_us.load(Ordering::Relaxed),
            avg_tick_us,
            execution_mode: if current.gate_config.force_supervised_mode {
                ExecutionMode::Supervised {
                    reason: SupervisionReason::ForcedSupervision,
                }
            } else if current.global_stats.is_well_calibrated {
                ExecutionMode::Autonomous
            } else {
                ExecutionMode::DryRun {
                    reason: DryRunReason::PoorCalibration,
                }
            },
            gate_open: current.global_stats.is_well_calibrated,
            signals,
            brier_score: current.global_stats.lifetime_brier,
            ece: current.global_stats.ece,
            predictions_made: current.global_stats.total_predictions as u64,
            predictions_resolved: current.global_stats.total_predictions as u64, // Same for now
            pending_predictions: pending.len(),
            current_prediction,
            recent_logs: logs.iter().cloned().collect(),
            domain_briers,
            calibration_quality: if current.global_stats.lifetime_brier < 0.10 {
                CalibrationQuality::Excellent
            } else if current.global_stats.lifetime_brier < 0.15 {
                CalibrationQuality::Good
            } else if current.global_stats.lifetime_brier < 0.25 {
                CalibrationQuality::Moderate
            } else if current.global_stats.total_predictions >= 10 {
                CalibrationQuality::Poor
            } else {
                CalibrationQuality::Insufficient
            },
            lifetime_predictions: current.global_stats.total_predictions as u64,
        }
    }

    /// Log a message
    fn log(&self, level: LogLevel, message: impl Into<String>) {
        let mut logs = self.logs.lock();
        let entry = RuntimeLogEntry {
            tick: self.tick_count.load(Ordering::Relaxed),
            timestamp: Instant::now(),
            level,
            message: message.into(),
        };

        if logs.len() >= self.config.log_buffer_size {
            logs.pop_front();
        }
        logs.push_back(entry);
    }

    /// Set runtime state
    fn set_state(&self, new_state: RuntimeState) {
        let mut state = self.state.write();
        if *state != new_state {
            *state = new_state;
            drop(state);

            if let Some(ref tx) = self.event_tx {
                let _ = tx.try_send(RuntimeEvent::StateChanged(new_state));
            }
        }
    }

    /// Queue a prediction for resolution
    pub fn queue_prediction(
        &self,
        prediction: WorldPrediction,
        auto_resolve: Option<AutoResolveType>,
    ) {
        let mut pending = self.pending.lock();
        if pending.len() < self.config.max_pending_predictions {
            let deadline = auto_resolve.as_ref().map(|ar| {
                Instant::now()
                    + match ar {
                        AutoResolveType::Timeout(d) => *d,
                        _ => Duration::from_secs(30),
                    }
            });

            pending.push_back(PendingPrediction {
                prediction,
                created_at: Instant::now(),
                deadline,
                auto_resolve,
            });
        }
    }

    /// Pause the runtime
    pub fn pause(&self) {
        self.set_state(RuntimeState::Paused);
        self.log(LogLevel::Info, "Runtime paused");
    }

    /// Resume the runtime
    pub fn resume(&self) {
        if self.state() == RuntimeState::Paused {
            self.set_state(RuntimeState::Running);
            self.log(LogLevel::Info, "Runtime resumed");
        }
    }

    /// Stop the runtime
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.set_state(RuntimeState::Stopping);
        self.log(LogLevel::Info, "Runtime stopping");
    }

    /// Run the main loop (async)
    pub async fn run(&self) -> anyhow::Result<()> {
        self.running.store(true, Ordering::SeqCst);
        self.set_state(RuntimeState::Running);
        self.log(LogLevel::Success, "MAGI Runtime started");

        let mut ticker = interval(Duration::from_millis(self.config.tick_interval_ms));

        while self.running.load(Ordering::SeqCst) {
            ticker.tick().await;

            // Skip if paused
            if self.state() == RuntimeState::Paused {
                continue;
            }

            // Execute one tick
            let tick_start = Instant::now();
            if let Err(e) = self.tick().await {
                self.log(LogLevel::Error, format!("Tick error: {}", e));
                if let Some(ref tx) = self.event_tx {
                    let _ = tx.try_send(RuntimeEvent::Error(e.to_string()));
                }
            }
            let tick_duration = tick_start.elapsed();

            // Update timing
            let tick_us = tick_duration.as_micros() as u64;
            self.last_tick_us.store(tick_us, Ordering::Relaxed);
            self.total_tick_us.fetch_add(tick_us, Ordering::Relaxed);
            self.tick_count.fetch_add(1, Ordering::Relaxed);
        }

        // Final persist
        let _ = self.model.lock().sync_and_save();
        self.log(LogLevel::Info, "Runtime stopped gracefully");

        Ok(())
    }

    /// Execute one tick of the cognitive cycle
    async fn tick(&self) -> anyhow::Result<()> {
        // 1. PULSE - We're awake

        // 2. PERCEIVE - Generate signals from bridge
        let signals = if self.config.enable_bridge_signals {
            let model = self.model.lock();
            let bridge = self.bridge.lock();
            let wgsm = model.model();
            let principled = bridge.generate_signals(wgsm);

            RuntimeSignals {
                prediction_error: principled.prediction_error,
                uncertainty: principled.uncertainty,
                confidence: principled.confidence,
                coherence: principled.coherence,
                free_energy: principled.prediction_error * 10.0 + principled.uncertainty * 5.0,
                surprise: if principled.prediction_error > 0.3 {
                    principled.prediction_error
                } else {
                    0.0
                },
            }
        } else {
            RuntimeSignals::default()
        };

        // Update cached signals
        *self.signals.write() = signals.clone();

        // 3. CHECK STATE - Should we be observing?
        if signals.uncertainty > self.config.observation_threshold {
            if self.state() != RuntimeState::Observing {
                self.set_state(RuntimeState::Observing);
                self.log(
                    LogLevel::Warning,
                    format!(
                        "Entering observation mode (uncertainty: {:.2})",
                        signals.uncertainty
                    ),
                );
            }
        } else if self.state() == RuntimeState::Observing {
            self.set_state(RuntimeState::Running);
            self.log(LogLevel::Success, "Exiting observation mode");
        }

        // 4. RESOLVE - Check pending predictions
        self.resolve_pending().await?;

        // 5. PERSIST - Save if needed
        let tick = self.tick_count.load(Ordering::Relaxed);
        if self.config.persist_interval_ticks == 0 || tick % self.config.persist_interval_ticks == 0
        {
            let _ = self.model.lock().sync_and_save();
        }

        Ok(())
    }

    /// Resolve pending predictions
    async fn resolve_pending(&self) -> anyhow::Result<()> {
        let mut pending = self.pending.lock();
        let mut resolved = Vec::new();

        for (idx, pred) in pending.iter().enumerate() {
            // Check if past deadline
            if let Some(deadline) = pred.deadline {
                if Instant::now() > deadline {
                    resolved.push((idx, false)); // Timeout = failure
                    continue;
                }
            }

            // Check auto-resolve
            if let Some(ref auto) = pred.auto_resolve {
                if let Some(success) = self.check_auto_resolve(auto).await {
                    resolved.push((idx, success));
                }
            }
        }

        // Process resolutions (in reverse to maintain indices)
        for (idx, success) in resolved.into_iter().rev() {
            if let Some(pred) = pending.remove(idx) {
                // Record resolution via the model
                let mut model = self.model.lock();

                // Resolve the prediction in the underlying WorldGroundedSelfModel
                let outcome = if success {
                    OutcomeCategory::Success
                } else {
                    OutcomeCategory::SafeFailure
                };
                let _ = model.model_mut().resolve_prediction(
                    &pred.prediction.id,
                    outcome,
                    1.0, // Full confidence in resolution
                );

                // Trigger persistence sync
                model.on_resolution();

                // Inform the bridge for PAC tracking
                {
                    let mut bridge = self.bridge.lock();
                    bridge.observe_resolution(pred.prediction.confidence, success);
                }

                // Calculate Brier score from the prediction
                let brier = pred.prediction.brier_score().unwrap_or(0.5);

                // Log
                let status = if success { "Success" } else { "Failure" };
                self.log(
                    if success {
                        LogLevel::Success
                    } else {
                        LogLevel::Warning
                    },
                    format!(
                        "Resolved: {} ({}). Brier: {:.4}",
                        pred.prediction.claim, status, brier
                    ),
                );

                // Emit event
                if let Some(ref tx) = self.event_tx {
                    let _ = tx.try_send(RuntimeEvent::PredictionResolved { success, brier });
                }
            }
        }

        Ok(())
    }

    /// Check if an auto-resolve condition is met
    async fn check_auto_resolve(&self, auto: &AutoResolveType) -> Option<bool> {
        match auto {
            AutoResolveType::FileExists(path) => Some(std::path::Path::new(path).exists()),
            AutoResolveType::CommandSucceeds(cmd) => {
                let output = tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(cmd)
                    .output()
                    .await
                    .ok()?;
                Some(output.status.success())
            }
            AutoResolveType::PortOpen(host, port) => {
                let addr = format!("{}:{}", host, port);
                let result = tokio::net::TcpStream::connect(&addr).await;
                Some(result.is_ok())
            }
            AutoResolveType::ServiceRunning(name) => {
                let output = tokio::process::Command::new("systemctl")
                    .args(["is-active", name])
                    .output()
                    .await
                    .ok()?;
                Some(output.status.success())
            }
            AutoResolveType::Timeout(_) => {
                // Timeouts are handled by deadline check
                None
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_config_defaults() {
        let config = RuntimeConfig::default();
        assert_eq!(config.tick_interval_ms, 1000);
        assert_eq!(config.max_pending_predictions, 100);
    }

    #[test]
    fn test_runtime_state_default() {
        assert_eq!(RuntimeState::default(), RuntimeState::Initializing);
    }

    #[tokio::test]
    async fn test_runtime_creation() {
        let config = RuntimeConfig {
            persistence_config: PersistenceConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let runtime = MagiLoopRuntime::new(config);
        assert!(runtime.is_ok());
    }

    #[tokio::test]
    async fn test_runtime_snapshot() {
        let config = RuntimeConfig {
            persistence_config: PersistenceConfig {
                enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let runtime = MagiLoopRuntime::new(config).unwrap();
        let snapshot = runtime.snapshot();
        assert_eq!(snapshot.state, RuntimeState::Initializing);
        assert_eq!(snapshot.tick_count, 0);
    }
}
