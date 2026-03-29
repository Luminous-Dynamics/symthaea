// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Autonomy Loop — Fabrication state machine with consciousness bridge.
//!
//! Tracks fabrication lifecycle (Idle → Printing → QC → Complete/Failed)
//! and emits lightweight event data that the caller wraps into full
//! `FabricationEvent`s with timestamps for the consciousness loop.

use serde::{Deserialize, Serialize};

// ── Event Bridge Types ────────────────────────────────────────────────

/// Lightweight event data for bridging to FabricationManager.
/// The caller wraps this into a full FabricationEvent with timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricationEventData {
    pub kind: FabricationEventDataKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FabricationEventDataKind {
    PrintStarted { job_id: String },
    PrintCompleted { job_id: String, quality_score: f32 },
    SafetyLevelChanged { level: String },
    QcPassed,
    QcFailed { reason: String },
}

// ── State Machine ─────────────────────────────────────────────────────

/// Fabrication lifecycle states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomyState {
    Idle,
    Printing { job_id: String },
    QualityCheck { job_id: String },
    Complete { job_id: String },
    Failed { reason: String },
}

/// Events that drive state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutonomyEvent {
    PrintStarted(String),
    PrintCompleted(f32),
    QcPassed,
    QcFailed(String),
    Error(String),
    Reset,
}

/// A recorded state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub from: AutonomyState,
    pub to: AutonomyState,
    pub event: AutonomyEvent,
}

/// Fabrication autonomy loop state machine.
///
/// Tracks the lifecycle of a fabrication job through discrete states
/// and records a history of transitions for event bridging.
pub struct AutonomyLoop {
    state: AutonomyState,
    history: Vec<HistoryEntry>,
}

impl AutonomyLoop {
    pub fn new() -> Self {
        Self {
            state: AutonomyState::Idle,
            history: Vec::new(),
        }
    }

    /// Current state.
    pub fn state(&self) -> &AutonomyState {
        &self.state
    }

    /// Full transition history.
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Apply an event, returning `true` if the transition was valid.
    pub fn apply(&mut self, event: AutonomyEvent) -> bool {
        let next = match (&self.state, &event) {
            (AutonomyState::Idle, AutonomyEvent::PrintStarted(job_id)) => {
                Some(AutonomyState::Printing {
                    job_id: job_id.clone(),
                })
            }
            (AutonomyState::Printing { job_id }, AutonomyEvent::PrintCompleted(_)) => {
                Some(AutonomyState::QualityCheck {
                    job_id: job_id.clone(),
                })
            }
            (AutonomyState::QualityCheck { job_id }, AutonomyEvent::QcPassed) => {
                Some(AutonomyState::Complete {
                    job_id: job_id.clone(),
                })
            }
            (AutonomyState::QualityCheck { .. }, AutonomyEvent::QcFailed(reason)) => {
                Some(AutonomyState::Failed {
                    reason: reason.clone(),
                })
            }
            (_, AutonomyEvent::Error(reason)) => Some(AutonomyState::Failed {
                reason: reason.clone(),
            }),
            (
                AutonomyState::Complete { .. } | AutonomyState::Failed { .. },
                AutonomyEvent::Reset,
            ) => Some(AutonomyState::Idle),
            _ => None,
        };

        if let Some(next_state) = next {
            let entry = HistoryEntry {
                from: self.state.clone(),
                to: next_state.clone(),
                event,
            };
            self.state = next_state;
            self.history.push(entry);
            true
        } else {
            false
        }
    }

    /// Convert the most recent state transition into FabricationEvents for the consciousness loop.
    ///
    /// Looks at the last history entry and maps state transitions to event data:
    /// - `PrintStarted(job_id)` → `FabricationEventDataKind::PrintStarted`
    /// - `PrintCompleted(score)` → `FabricationEventDataKind::PrintCompleted`
    /// - `QcPassed` → `FabricationEventDataKind::QcPassed`
    /// - `QcFailed(reason)` → `FabricationEventDataKind::QcFailed`
    /// - `Error(_)` when state is `Failed` → `SafetyLevelChanged("Red")`
    /// - Everything else → empty vec
    pub fn to_fabrication_events(&self) -> Vec<FabricationEventData> {
        let entry = match self.history.last() {
            Some(e) => e,
            None => return Vec::new(),
        };

        match &entry.event {
            AutonomyEvent::PrintStarted(job_id) => {
                vec![FabricationEventData {
                    kind: FabricationEventDataKind::PrintStarted {
                        job_id: job_id.clone(),
                    },
                }]
            }
            AutonomyEvent::PrintCompleted(score) => {
                // Extract job_id from the source state (Printing { job_id }).
                let job_id = match &entry.from {
                    AutonomyState::Printing { job_id } => job_id.clone(),
                    _ => String::new(),
                };
                vec![FabricationEventData {
                    kind: FabricationEventDataKind::PrintCompleted {
                        job_id,
                        quality_score: *score,
                    },
                }]
            }
            AutonomyEvent::QcPassed => {
                vec![FabricationEventData {
                    kind: FabricationEventDataKind::QcPassed,
                }]
            }
            AutonomyEvent::QcFailed(reason) => {
                vec![FabricationEventData {
                    kind: FabricationEventDataKind::QcFailed {
                        reason: reason.clone(),
                    },
                }]
            }
            AutonomyEvent::Error(_) => {
                if matches!(entry.to, AutonomyState::Failed { .. }) {
                    vec![FabricationEventData {
                        kind: FabricationEventDataKind::SafetyLevelChanged {
                            level: "Red".to_string(),
                        },
                    }]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }
}

impl Default for AutonomyLoop {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_started_emits_event() {
        let mut loop_ = AutonomyLoop::new();
        assert!(loop_.apply(AutonomyEvent::PrintStarted("job-42".into())));

        let events = loop_.to_fabrication_events();
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            FabricationEventDataKind::PrintStarted { job_id } => {
                assert_eq!(job_id, "job-42");
            }
            other => panic!("Expected PrintStarted, got {:?}", other),
        }
    }

    #[test]
    fn test_print_completed_emits_event() {
        let mut loop_ = AutonomyLoop::new();
        loop_.apply(AutonomyEvent::PrintStarted("job-99".into()));
        assert!(loop_.apply(AutonomyEvent::PrintCompleted(0.95)));

        let events = loop_.to_fabrication_events();
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            FabricationEventDataKind::PrintCompleted {
                job_id,
                quality_score,
            } => {
                assert_eq!(job_id, "job-99");
                assert!((quality_score - 0.95).abs() < 1e-6);
            }
            other => panic!("Expected PrintCompleted, got {:?}", other),
        }
    }

    #[test]
    fn test_qc_failed_emits_event() {
        let mut loop_ = AutonomyLoop::new();
        loop_.apply(AutonomyEvent::PrintStarted("job-1".into()));
        loop_.apply(AutonomyEvent::PrintCompleted(0.5));
        assert!(loop_.apply(AutonomyEvent::QcFailed("layer delamination".into())));

        let events = loop_.to_fabrication_events();
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            FabricationEventDataKind::QcFailed { reason } => {
                assert_eq!(reason, "layer delamination");
            }
            other => panic!("Expected QcFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_failure_emits_safety_event() {
        let mut loop_ = AutonomyLoop::new();
        loop_.apply(AutonomyEvent::PrintStarted("job-1".into()));
        assert!(loop_.apply(AutonomyEvent::Error("thermal runaway".into())));

        assert!(matches!(loop_.state(), AutonomyState::Failed { .. }));
        let events = loop_.to_fabrication_events();
        assert_eq!(events.len(), 1);
        match &events[0].kind {
            FabricationEventDataKind::SafetyLevelChanged { level } => {
                assert_eq!(level, "Red");
            }
            other => panic!("Expected SafetyLevelChanged, got {:?}", other),
        }
    }

    #[test]
    fn test_idle_no_events() {
        let loop_ = AutonomyLoop::new();
        let events = loop_.to_fabrication_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_reset_no_events() {
        let mut loop_ = AutonomyLoop::new();
        loop_.apply(AutonomyEvent::PrintStarted("job-1".into()));
        loop_.apply(AutonomyEvent::PrintCompleted(0.9));
        loop_.apply(AutonomyEvent::QcPassed);
        assert!(loop_.apply(AutonomyEvent::Reset));
        assert_eq!(*loop_.state(), AutonomyState::Idle);
        // Reset produces no fabrication events
        let events = loop_.to_fabrication_events();
        assert!(events.is_empty());
    }

    #[test]
    fn test_invalid_transition_rejected() {
        let mut loop_ = AutonomyLoop::new();
        // Cannot complete a print from Idle
        assert!(!loop_.apply(AutonomyEvent::PrintCompleted(1.0)));
        assert_eq!(*loop_.state(), AutonomyState::Idle);
    }

    #[test]
    fn test_qc_passed_emits_event() {
        let mut loop_ = AutonomyLoop::new();
        loop_.apply(AutonomyEvent::PrintStarted("job-7".into()));
        loop_.apply(AutonomyEvent::PrintCompleted(0.98));
        assert!(loop_.apply(AutonomyEvent::QcPassed));

        let events = loop_.to_fabrication_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].kind, FabricationEventDataKind::QcPassed));
    }
}
