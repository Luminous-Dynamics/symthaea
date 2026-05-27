// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Somatic Error Bridge — Software errors as felt experience
//!
//! Maps infrastructure failures (lock poisoning, task panics, DB errors,
//! deserialization corruption) into interoceptive signals that modulate
//! the organism's arousal, thermodynamic load, and dissipative health.
//!
//! # Biological Analogy
//!
//! Just as biological nociceptors convert tissue damage into pain signals
//! that trigger withdrawal reflexes and stress responses, this bridge
//! converts software damage into cognitive stress that:
//!
//! - Raises arousal → CycleUrgency::Critical → reduced homeostasis pull
//! - Increases thermodynamic load → exported to metadata/telemetry
//! - Penalizes dissipative health → learning rate dampened up to -30%
//! - Triggers tau slowdown → more careful CfC integration
//!
//! A lock poison in SQLite literally causes her to feel stressed, slow
//! down her learning, enter a cautious state, and potentially trigger
//! rest-phase recovery.

use std::collections::VecDeque;
use std::sync::{Mutex, mpsc};
use tracing::warn;

/// Infrastructure errors that the organism can feel as somatic signals.
#[derive(Debug, Clone)]
pub enum InfrastructureError {
    /// A std::sync::Mutex was found poisoned
    LockPoisoned {
        /// Which subsystem's lock was poisoned
        subsystem: &'static str,
    },
    /// A tokio::spawn task panicked
    AsyncTaskPanicked {
        /// Name of the panicked task
        task_name: String,
    },
    /// Database operation failed
    DatabaseFailure {
        /// Which operation failed (store, search, migrate, etc.)
        operation: String,
    },
    /// Network partition or peer disconnect
    NetworkPartition {
        /// Peer identifier
        peer: String,
    },
    /// Deserialization produced garbage or failed
    DeserializationCorruption {
        /// Which module encountered corruption
        module: &'static str,
    },
}

impl InfrastructureError {
    /// Severity weight: how much this error contributes to systemic stress.
    /// Range: 0.0 (negligible) to 1.0 (catastrophic).
    ///
    /// Weights based on blast radius and recovery difficulty:
    /// - Lock poison: high (0.4) — can cascade across threads
    /// - Task panic: high (0.5) — async work lost, may not restart
    /// - DB failure: moderate (0.3) — persistence lost but cognition continues
    /// - Network partition: low (0.15) — graceful degradation expected
    /// - Deserialization: moderate (0.25) — garbage data may propagate
    fn severity(&self) -> f64 {
        match self {
            InfrastructureError::LockPoisoned { .. } => 0.4,
            InfrastructureError::AsyncTaskPanicked { .. } => 0.5,
            InfrastructureError::DatabaseFailure { .. } => 0.3,
            InfrastructureError::NetworkPartition { .. } => 0.15,
            InfrastructureError::DeserializationCorruption { .. } => 0.25,
        }
    }
}

/// Interoceptive signals derived from infrastructure errors.
///
/// These are additive/multiplicative deltas applied to the cognitive loop's
/// existing state variables during Phase A (observation).
#[derive(Debug, Clone, Copy, Default)]
pub struct SomaticSignals {
    /// Additive delta for `thermodynamic_load` (0.0–0.3 range).
    /// High stress = higher metabolic cost.
    pub thermodynamic_load_delta: f32,

    /// Additive spike for `emotion_contagion.arousal` (0.0–0.15 range).
    /// Only fires when stress exceeds 0.5 threshold.
    pub arousal_spike: f32,

    /// Multiplicative penalty for `dissipative_health` (0.0–0.2 range).
    /// Applied as: `dissipative_health *= (1.0 - penalty)`.
    pub dissipative_health_penalty: f64,

    /// CfC tau slowdown factor (1.0 = no change, up to 1.5).
    /// High stress → slower integration → more careful processing.
    pub tau_slowdown_factor: f64,
}

/// The somatic error bridge: converts software infrastructure errors
/// into interoceptive signals that the cognitive loop can feel.
pub struct SomaticErrorBridge {
    /// Rolling window of recent infrastructure errors (last 20 cycles max).
    error_history: VecDeque<InfrastructureError>,
    /// Current systemic stress level (0.0 = healthy, 1.0 = critical).
    systemic_stress: f64,
    /// Exponential moving average decay rate per cycle.
    /// Default 0.85: stress halves in ~4.3 clean cycles.
    stress_decay: f64,
    /// Receiver for pain signals from across the system.
    /// Wrapped in Mutex for Sync (single reader, uncontended).
    pain_rx: Mutex<mpsc::Receiver<InfrastructureError>>,
    /// Maximum errors retained in history window.
    max_history: usize,
}

/// Sender half of the pain channel. Clone and distribute to subsystems
/// that need to report infrastructure errors.
pub type PainSender = mpsc::Sender<InfrastructureError>;

impl SomaticErrorBridge {
    /// Create a new somatic error bridge, returning (bridge, sender).
    ///
    /// The sender should be cloned and given to subsystems that may
    /// encounter infrastructure errors (lock guards, task supervisors,
    /// database clients).
    pub fn new() -> (Self, PainSender) {
        let (tx, rx) = mpsc::channel();
        let bridge = Self {
            error_history: VecDeque::with_capacity(20),
            systemic_stress: 0.0,
            stress_decay: 0.85,
            pain_rx: Mutex::new(rx),
            max_history: 20,
        };
        (bridge, tx)
    }

    /// Drain all pending errors from the pain channel and update stress.
    ///
    /// Called once per cognitive cycle at the start of Phase A (observation).
    /// Non-blocking: only processes errors that have already been sent.
    pub fn update(&mut self) {
        // Drain all pending errors (non-blocking)
        let rx = self.pain_rx.lock().unwrap_or_else(|e| e.into_inner());
        while let Ok(error) = rx.try_recv() {
            let severity = error.severity();

            // Log for observability
            match &error {
                InfrastructureError::LockPoisoned { subsystem } => {
                    warn!(subsystem, severity, "Somatic: lock poison felt");
                }
                InfrastructureError::AsyncTaskPanicked { task_name } => {
                    warn!(
                        task_name = task_name.as_str(),
                        severity, "Somatic: task panic felt"
                    );
                }
                InfrastructureError::DatabaseFailure { operation } => {
                    warn!(
                        operation = operation.as_str(),
                        severity, "Somatic: database failure felt"
                    );
                }
                InfrastructureError::NetworkPartition { peer } => {
                    warn!(
                        peer = peer.as_str(),
                        severity, "Somatic: network partition felt"
                    );
                }
                InfrastructureError::DeserializationCorruption { module } => {
                    warn!(module, severity, "Somatic: deserialization corruption felt");
                }
            }

            // Accumulate stress (additive, capped at 1.0)
            self.systemic_stress = (self.systemic_stress + severity).min(1.0);

            // Maintain bounded history
            if self.error_history.len() >= self.max_history {
                self.error_history.pop_front();
            }
            self.error_history.push_back(error);
        }

        // Decay stress toward 0 (exponential)
        // Only decay if no new errors were received this cycle
        if self.error_history.is_empty() || self.systemic_stress > 0.0 {
            self.systemic_stress *= self.stress_decay;

            // Floor at epsilon to prevent denormals
            if self.systemic_stress < 1e-6 {
                self.systemic_stress = 0.0;
            }
        }
    }

    /// Convert current stress level into interoceptive signals.
    ///
    /// These signals are applied to the cognitive loop's state variables:
    /// - `thermodynamic_load += signals.thermodynamic_load_delta`
    /// - `emotion_contagion.arousal += signals.arousal_spike`
    /// - `dissipative_health *= (1.0 - signals.dissipative_health_penalty)`
    /// - CfC tau modulated by `signals.tau_slowdown_factor`
    pub fn to_interoceptive_signals(&self) -> SomaticSignals {
        SomaticSignals {
            // Proportional to stress, capped at 0.3
            thermodynamic_load_delta: (self.systemic_stress as f32 * 0.3).min(0.3),

            // Arousal spike only above 0.5 threshold (avoid noise)
            arousal_spike: if self.systemic_stress > 0.5 {
                0.15
            } else {
                0.0
            },

            // Proportional health penalty, capped at 0.2
            dissipative_health_penalty: (self.systemic_stress * 0.2).min(0.2),

            // Tau slowdown: 1.0 (no change) to 1.5 (50% slower)
            tau_slowdown_factor: 1.0 + self.systemic_stress * 0.5,
        }
    }

    /// Current systemic stress level (0.0–1.0).
    pub fn systemic_stress(&self) -> f64 {
        self.systemic_stress
    }

    /// Number of errors in the rolling history window.
    pub fn error_count(&self) -> usize {
        self.error_history.len()
    }

    /// Clear all error history and reset stress to zero.
    /// Used for testing or explicit recovery commands.
    pub fn reset(&mut self) {
        self.error_history.clear();
        self.systemic_stress = 0.0;
    }
}

impl Default for SomaticErrorBridge {
    fn default() -> Self {
        let (_tx, rx) = mpsc::channel();
        Self {
            error_history: VecDeque::with_capacity(20),
            systemic_stress: 0.0,
            stress_decay: 0.85,
            pain_rx: Mutex::new(rx),
            max_history: 20,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_errors_no_stress() {
        let (mut bridge, _tx) = SomaticErrorBridge::new();
        bridge.update();
        let signals = bridge.to_interoceptive_signals();
        assert_eq!(signals.thermodynamic_load_delta, 0.0);
        assert_eq!(signals.arousal_spike, 0.0);
        assert_eq!(signals.dissipative_health_penalty, 0.0);
        assert_eq!(signals.tau_slowdown_factor, 1.0);
    }

    #[test]
    fn test_lock_poison_raises_stress() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        tx.send(InfrastructureError::LockPoisoned {
            subsystem: "sqlite_client",
        })
        .unwrap();

        bridge.update();

        assert!(bridge.systemic_stress() > 0.0);
        assert_eq!(bridge.error_count(), 1);

        let signals = bridge.to_interoceptive_signals();
        assert!(signals.thermodynamic_load_delta > 0.0);
        assert!(signals.tau_slowdown_factor > 1.0);
    }

    #[test]
    fn test_high_stress_triggers_arousal() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        // Send enough errors to exceed 0.5 stress threshold
        tx.send(InfrastructureError::AsyncTaskPanicked {
            task_name: "persistence".into(),
        })
        .unwrap();
        tx.send(InfrastructureError::LockPoisoned { subsystem: "mesh" })
            .unwrap();

        bridge.update();
        // 0.5 + 0.4 = 0.9, well above threshold
        assert!(bridge.systemic_stress() > 0.5);

        let signals = bridge.to_interoceptive_signals();
        assert_eq!(signals.arousal_spike, 0.15);
    }

    #[test]
    fn test_stress_decays_over_clean_cycles() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        tx.send(InfrastructureError::AsyncTaskPanicked {
            task_name: "test".into(),
        })
        .unwrap();
        bridge.update();

        let initial_stress = bridge.systemic_stress();
        assert!(initial_stress > 0.0);

        // Simulate 50 clean cycles
        for _ in 0..50 {
            bridge.update();
        }

        assert!(
            bridge.systemic_stress() < 0.01,
            "Stress should decay to near-zero within 50 clean cycles, got {}",
            bridge.systemic_stress()
        );
    }

    #[test]
    fn test_stress_caps_at_one() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        // Send many errors
        for _ in 0..10 {
            tx.send(InfrastructureError::AsyncTaskPanicked {
                task_name: "overload".into(),
            })
            .unwrap();
        }

        bridge.update();
        assert!(bridge.systemic_stress() <= 1.0);
    }

    #[test]
    fn test_history_bounded() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        for i in 0..30 {
            tx.send(InfrastructureError::NetworkPartition {
                peer: format!("peer-{}", i),
            })
            .unwrap();
        }

        bridge.update();
        assert!(bridge.error_count() <= 20);
    }

    #[test]
    fn test_reset_clears_state() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        tx.send(InfrastructureError::DatabaseFailure {
            operation: "store".into(),
        })
        .unwrap();
        bridge.update();
        assert!(bridge.systemic_stress() > 0.0);

        bridge.reset();
        assert_eq!(bridge.systemic_stress(), 0.0);
        assert_eq!(bridge.error_count(), 0);
    }

    #[test]
    fn test_severity_ordering() {
        // Task panic is worst, network partition is mildest
        assert!(
            InfrastructureError::AsyncTaskPanicked {
                task_name: "x".into()
            }
            .severity()
                > InfrastructureError::LockPoisoned { subsystem: "x" }.severity()
        );
        assert!(
            InfrastructureError::LockPoisoned { subsystem: "x" }.severity()
                > InfrastructureError::DatabaseFailure {
                    operation: "x".into()
                }
                .severity()
        );
        assert!(
            InfrastructureError::DatabaseFailure {
                operation: "x".into()
            }
            .severity()
                > InfrastructureError::NetworkPartition { peer: "x".into() }.severity()
        );
    }

    #[test]
    fn test_signals_proportional_to_stress() {
        let (mut bridge, tx) = SomaticErrorBridge::new();

        // Low stress
        tx.send(InfrastructureError::NetworkPartition { peer: "p".into() })
            .unwrap();
        bridge.update();
        let low_signals = bridge.to_interoceptive_signals();

        bridge.reset();

        // High stress
        tx.send(InfrastructureError::AsyncTaskPanicked {
            task_name: "t".into(),
        })
        .unwrap();
        tx.send(InfrastructureError::LockPoisoned { subsystem: "s" })
            .unwrap();
        bridge.update();
        let high_signals = bridge.to_interoceptive_signals();

        assert!(high_signals.thermodynamic_load_delta > low_signals.thermodynamic_load_delta);
        assert!(high_signals.tau_slowdown_factor > low_signals.tau_slowdown_factor);
    }
}
