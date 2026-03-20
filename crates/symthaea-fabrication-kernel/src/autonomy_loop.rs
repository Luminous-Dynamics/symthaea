//! End-to-end manufacturing autonomy state machine
//!
//! Models the complete order lifecycle: receive → design → DFM check →
//! MRP planning → print → quality check → ship → payment → provenance → done.
//! Every transition is validated; invalid events are rejected with
//! [`AutonomyError::InvalidTransition`].

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Current phase of the manufacturing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutonomyState {
    /// Idle — no active order.
    Idle,
    /// An order has been received and is awaiting design.
    OrderReceived {
        order_id: String,
    },
    /// Generative or manual design is in progress.
    Designing {
        order_id: String,
        design_hash: Option<String>,
    },
    /// Design-for-manufacturability check.
    DfmCheck {
        order_id: String,
        design_hash: String,
    },
    /// Material Requirements Planning pass.
    MrpPlanning {
        order_id: String,
        design_hash: String,
    },
    /// Additive or subtractive manufacturing in progress.
    Printing {
        order_id: String,
        job_id: String,
    },
    /// Post-print quality inspection.
    QualityCheck {
        order_id: String,
        job_id: String,
        quality_score: Option<f32>,
    },
    /// Packaged and being shipped.
    Shipping {
        order_id: String,
    },
    /// Awaiting payment settlement.
    PaymentSettlement {
        order_id: String,
        amount: u64,
    },
    /// Recording provenance on-chain.
    ProvenanceRecording {
        order_id: String,
    },
    /// Successfully completed.
    Complete {
        order_id: String,
    },
    /// Terminal failure.
    Failed {
        order_id: String,
        reason: String,
    },
}

impl AutonomyState {
    /// Extract the order_id from any variant that carries one.
    pub fn order_id(&self) -> Option<&str> {
        match self {
            Self::Idle => None,
            Self::OrderReceived { order_id }
            | Self::Designing { order_id, .. }
            | Self::DfmCheck { order_id, .. }
            | Self::MrpPlanning { order_id, .. }
            | Self::Printing { order_id, .. }
            | Self::QualityCheck { order_id, .. }
            | Self::Shipping { order_id }
            | Self::PaymentSettlement { order_id, .. }
            | Self::ProvenanceRecording { order_id }
            | Self::Complete { order_id }
            | Self::Failed { order_id, .. } => Some(order_id),
        }
    }

    /// Short label for logging / display.
    fn label(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::OrderReceived { .. } => "OrderReceived",
            Self::Designing { .. } => "Designing",
            Self::DfmCheck { .. } => "DfmCheck",
            Self::MrpPlanning { .. } => "MrpPlanning",
            Self::Printing { .. } => "Printing",
            Self::QualityCheck { .. } => "QualityCheck",
            Self::Shipping { .. } => "Shipping",
            Self::PaymentSettlement { .. } => "PaymentSettlement",
            Self::ProvenanceRecording { .. } => "ProvenanceRecording",
            Self::Complete { .. } => "Complete",
            Self::Failed { .. } => "Failed",
        }
    }
}

impl fmt::Display for AutonomyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::OrderReceived { order_id } => write!(f, "OrderReceived({order_id})"),
            Self::Designing { order_id, .. } => write!(f, "Designing({order_id})"),
            Self::DfmCheck { order_id, .. } => write!(f, "DfmCheck({order_id})"),
            Self::MrpPlanning { order_id, .. } => write!(f, "MrpPlanning({order_id})"),
            Self::Printing { order_id, job_id } => {
                write!(f, "Printing({order_id}, job={job_id})")
            }
            Self::QualityCheck { order_id, .. } => write!(f, "QualityCheck({order_id})"),
            Self::Shipping { order_id } => write!(f, "Shipping({order_id})"),
            Self::PaymentSettlement { order_id, amount } => {
                write!(f, "PaymentSettlement({order_id}, {amount})")
            }
            Self::ProvenanceRecording { order_id } => {
                write!(f, "ProvenanceRecording({order_id})")
            }
            Self::Complete { order_id } => write!(f, "Complete({order_id})"),
            Self::Failed { order_id, reason } => {
                write!(f, "Failed({order_id}: {reason})")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// External events that drive state transitions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutonomyEvent {
    /// A new manufacturing order has been placed.
    OrderPlaced,
    /// A design has been selected / generated.
    DesignSelected,
    /// DFM check passed.
    DfmPassed,
    /// DFM check failed with a reason.
    DfmFailed(String),
    /// MRP planning is complete.
    MrpComplete,
    /// Print job has started with the given job ID.
    PrintStarted(String),
    /// Print completed with a quality score.
    PrintCompleted(f32),
    /// QC inspection passed.
    QcPassed,
    /// QC failed with a reason.
    QcFailed(String),
    /// Product has been shipped.
    Shipped,
    /// Payment has been received.
    PaymentReceived,
    /// Provenance has been recorded on-chain.
    ProvenanceRecorded,
    /// A hard error occurred.
    Error(String),
}

impl fmt::Display for AutonomyEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrderPlaced => write!(f, "OrderPlaced"),
            Self::DesignSelected => write!(f, "DesignSelected"),
            Self::DfmPassed => write!(f, "DfmPassed"),
            Self::DfmFailed(r) => write!(f, "DfmFailed({r})"),
            Self::MrpComplete => write!(f, "MrpComplete"),
            Self::PrintStarted(j) => write!(f, "PrintStarted({j})"),
            Self::PrintCompleted(q) => write!(f, "PrintCompleted({q})"),
            Self::QcPassed => write!(f, "QcPassed"),
            Self::QcFailed(r) => write!(f, "QcFailed({r})"),
            Self::Shipped => write!(f, "Shipped"),
            Self::PaymentReceived => write!(f, "PaymentReceived"),
            Self::ProvenanceRecorded => write!(f, "ProvenanceRecorded"),
            Self::Error(e) => write!(f, "Error({e})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the autonomy loop.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AutonomyError {
    /// The event is not valid in the current state.
    InvalidTransition {
        from: String,
        event: String,
    },
    /// Referenced order does not match current state.
    OrderNotFound,
    /// QC score is below the configured threshold.
    QualityBelowThreshold,
}

impl fmt::Display for AutonomyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { from, event } => {
                write!(f, "invalid transition: {event} in state {from}")
            }
            Self::OrderNotFound => write!(f, "order not found"),
            Self::QualityBelowThreshold => write!(f, "quality below threshold"),
        }
    }
}

impl std::error::Error for AutonomyError {}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the autonomy loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyConfig {
    /// Minimum acceptable quality score (0.0–1.0).
    pub min_quality_score: f32,
    /// Whether to automatically retry on QC failure (back to Designing).
    pub auto_retry: bool,
    /// Maximum number of retries before failing permanently.
    pub max_retries: u32,
}

impl Default for AutonomyConfig {
    fn default() -> Self {
        Self {
            min_quality_score: 0.8,
            auto_retry: true,
            max_retries: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// AutonomyLoop
// ---------------------------------------------------------------------------

/// The manufacturing state machine.
pub struct AutonomyLoop {
    state: AutonomyState,
    history: Vec<(AutonomyState, AutonomyEvent, u64)>,
    config: AutonomyConfig,
    retry_count: u32,
}

impl AutonomyLoop {
    /// Create a new idle autonomy loop.
    pub fn new(config: AutonomyConfig) -> Self {
        Self {
            state: AutonomyState::Idle,
            history: Vec::new(),
            config,
            retry_count: 0,
        }
    }

    /// Process an event, returning the new state or an error.
    pub fn process_event(
        &mut self,
        event: AutonomyEvent,
        timestamp: u64,
    ) -> Result<&AutonomyState, AutonomyError> {
        // Error events are always accepted (transition to Failed).
        if let AutonomyEvent::Error(ref reason) = event {
            let order_id = self
                .state
                .order_id()
                .unwrap_or("unknown")
                .to_string();
            let prev = std::mem::replace(
                &mut self.state,
                AutonomyState::Failed {
                    order_id,
                    reason: reason.clone(),
                },
            );
            self.history.push((prev, event, timestamp));
            return Ok(&self.state);
        }

        let new_state = self.compute_transition(&event)?;
        let prev = std::mem::replace(&mut self.state, new_state);
        self.history.push((prev, event, timestamp));
        Ok(&self.state)
    }

    /// Current state.
    pub fn state(&self) -> &AutonomyState {
        &self.state
    }

    /// Whether the loop has reached a terminal state (Complete or Failed).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            AutonomyState::Complete { .. } | AutonomyState::Failed { .. }
        )
    }

    /// Full transition history.
    pub fn history(&self) -> &[(AutonomyState, AutonomyEvent, u64)] {
        &self.history
    }

    // -----------------------------------------------------------------------
    // Transition logic
    // -----------------------------------------------------------------------

    fn compute_transition(
        &mut self,
        event: &AutonomyEvent,
    ) -> Result<AutonomyState, AutonomyError> {
        match (&self.state, event) {
            // Idle → OrderReceived
            (AutonomyState::Idle, AutonomyEvent::OrderPlaced) => {
                let order_id = format!("ORD-{}", self.history.len() + 1);
                self.retry_count = 0;
                Ok(AutonomyState::OrderReceived { order_id })
            }

            // OrderReceived → Designing
            (AutonomyState::OrderReceived { order_id }, AutonomyEvent::DesignSelected) => {
                Ok(AutonomyState::Designing {
                    order_id: order_id.clone(),
                    design_hash: None,
                })
            }

            // Designing → DfmCheck
            (AutonomyState::Designing { order_id, .. }, AutonomyEvent::DfmPassed) => {
                let hash = format!("HASH-{}", timestamp_hash(order_id));
                Ok(AutonomyState::DfmCheck {
                    order_id: order_id.clone(),
                    design_hash: hash,
                })
            }

            // Designing → Failed (DFM caught early)
            (AutonomyState::Designing { order_id, .. }, AutonomyEvent::DfmFailed(reason)) => {
                Ok(AutonomyState::Failed {
                    order_id: order_id.clone(),
                    reason: reason.clone(),
                })
            }

            // DfmCheck → MrpPlanning (passed)
            (AutonomyState::DfmCheck { order_id, design_hash }, AutonomyEvent::MrpComplete) => {
                Ok(AutonomyState::MrpPlanning {
                    order_id: order_id.clone(),
                    design_hash: design_hash.clone(),
                })
            }

            // DfmCheck → Failed
            (AutonomyState::DfmCheck { order_id, .. }, AutonomyEvent::DfmFailed(reason)) => {
                Ok(AutonomyState::Failed {
                    order_id: order_id.clone(),
                    reason: reason.clone(),
                })
            }

            // MrpPlanning → Printing
            (
                AutonomyState::MrpPlanning { order_id, .. },
                AutonomyEvent::PrintStarted(job_id),
            ) => Ok(AutonomyState::Printing {
                order_id: order_id.clone(),
                job_id: job_id.clone(),
            }),

            // Printing → QualityCheck
            (
                AutonomyState::Printing { order_id, job_id },
                AutonomyEvent::PrintCompleted(score),
            ) => Ok(AutonomyState::QualityCheck {
                order_id: order_id.clone(),
                job_id: job_id.clone(),
                quality_score: Some(*score),
            }),

            // QualityCheck → Shipping (passed)
            (
                AutonomyState::QualityCheck {
                    order_id,
                    quality_score,
                    ..
                },
                AutonomyEvent::QcPassed,
            ) => {
                if let Some(score) = quality_score {
                    if *score < self.config.min_quality_score {
                        return Err(AutonomyError::QualityBelowThreshold);
                    }
                }
                Ok(AutonomyState::Shipping {
                    order_id: order_id.clone(),
                })
            }

            // QualityCheck → Designing (retry) or Failed
            (
                AutonomyState::QualityCheck { order_id, .. },
                AutonomyEvent::QcFailed(reason),
            ) => {
                if self.config.auto_retry && self.retry_count < self.config.max_retries {
                    self.retry_count += 1;
                    Ok(AutonomyState::Designing {
                        order_id: order_id.clone(),
                        design_hash: None,
                    })
                } else {
                    Ok(AutonomyState::Failed {
                        order_id: order_id.clone(),
                        reason: format!("QC failed after retries: {reason}"),
                    })
                }
            }

            // Shipping → PaymentSettlement
            (AutonomyState::Shipping { order_id }, AutonomyEvent::Shipped) => {
                Ok(AutonomyState::PaymentSettlement {
                    order_id: order_id.clone(),
                    amount: 0, // amount set by external system
                })
            }

            // PaymentSettlement → ProvenanceRecording
            (
                AutonomyState::PaymentSettlement { order_id, .. },
                AutonomyEvent::PaymentReceived,
            ) => Ok(AutonomyState::ProvenanceRecording {
                order_id: order_id.clone(),
            }),

            // ProvenanceRecording → Complete
            (
                AutonomyState::ProvenanceRecording { order_id },
                AutonomyEvent::ProvenanceRecorded,
            ) => Ok(AutonomyState::Complete {
                order_id: order_id.clone(),
            }),

            // Everything else is invalid.
            _ => Err(AutonomyError::InvalidTransition {
                from: self.state.label().to_string(),
                event: format!("{event}"),
            }),
        }
    }
}

/// Simple hash for generating design hashes.
fn timestamp_hash(s: &str) -> u64 {
    let mut h: u64 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u64);
    }
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_loop() -> AutonomyLoop {
        AutonomyLoop::new(AutonomyConfig::default())
    }

    /// Drive through the entire happy path.
    fn drive_happy_path(al: &mut AutonomyLoop) {
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        al.process_event(AutonomyEvent::PrintCompleted(0.95), 6)
            .unwrap();
        al.process_event(AutonomyEvent::QcPassed, 7).unwrap();
        al.process_event(AutonomyEvent::Shipped, 8).unwrap();
        al.process_event(AutonomyEvent::PaymentReceived, 9).unwrap();
        al.process_event(AutonomyEvent::ProvenanceRecorded, 10)
            .unwrap();
    }

    #[test]
    fn happy_path_end_to_end() {
        let mut al = default_loop();
        drive_happy_path(&mut al);
        assert!(matches!(al.state(), AutonomyState::Complete { .. }));
        assert!(al.is_terminal());
    }

    #[test]
    fn starts_idle() {
        let al = default_loop();
        assert!(matches!(al.state(), AutonomyState::Idle));
        assert!(!al.is_terminal());
    }

    #[test]
    fn invalid_transition_from_idle() {
        let mut al = default_loop();
        let res = al.process_event(AutonomyEvent::DesignSelected, 1);
        assert!(matches!(
            res,
            Err(AutonomyError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn invalid_transition_from_printing() {
        let mut al = default_loop();
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        // Can't ship from Printing.
        let res = al.process_event(AutonomyEvent::Shipped, 6);
        assert!(matches!(
            res,
            Err(AutonomyError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn dfm_failure_goes_to_failed() {
        let mut al = default_loop();
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmFailed("wall too thin".into()), 3)
            .unwrap();
        assert!(matches!(al.state(), AutonomyState::Failed { .. }));
        assert!(al.is_terminal());
    }

    #[test]
    fn qc_failure_with_retry() {
        let mut al = AutonomyLoop::new(AutonomyConfig {
            min_quality_score: 0.8,
            auto_retry: true,
            max_retries: 2,
        });
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        al.process_event(AutonomyEvent::PrintCompleted(0.5), 6)
            .unwrap();
        // QC fails — should retry (go back to Designing).
        al.process_event(AutonomyEvent::QcFailed("delamination".into()), 7)
            .unwrap();
        assert!(matches!(al.state(), AutonomyState::Designing { .. }));
        assert!(!al.is_terminal());
    }

    #[test]
    fn qc_failure_exhausts_retries() {
        let mut al = AutonomyLoop::new(AutonomyConfig {
            min_quality_score: 0.8,
            auto_retry: true,
            max_retries: 0,
        });
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        al.process_event(AutonomyEvent::PrintCompleted(0.5), 6)
            .unwrap();
        al.process_event(AutonomyEvent::QcFailed("warping".into()), 7)
            .unwrap();
        assert!(matches!(al.state(), AutonomyState::Failed { .. }));
    }

    #[test]
    fn qc_failure_no_retry_config() {
        let mut al = AutonomyLoop::new(AutonomyConfig {
            min_quality_score: 0.8,
            auto_retry: false,
            max_retries: 5,
        });
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        al.process_event(AutonomyEvent::PrintCompleted(0.5), 6)
            .unwrap();
        al.process_event(AutonomyEvent::QcFailed("cracks".into()), 7)
            .unwrap();
        assert!(matches!(al.state(), AutonomyState::Failed { .. }));
    }

    #[test]
    fn quality_below_threshold_rejects_qc_passed() {
        let mut al = default_loop();
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::DfmPassed, 3).unwrap();
        al.process_event(AutonomyEvent::MrpComplete, 4).unwrap();
        al.process_event(AutonomyEvent::PrintStarted("J1".into()), 5)
            .unwrap();
        // Score well below threshold.
        al.process_event(AutonomyEvent::PrintCompleted(0.3), 6)
            .unwrap();
        let res = al.process_event(AutonomyEvent::QcPassed, 7);
        assert!(matches!(res, Err(AutonomyError::QualityBelowThreshold)));
    }

    #[test]
    fn terminal_state_detection() {
        let mut al = default_loop();
        assert!(!al.is_terminal());
        drive_happy_path(&mut al);
        assert!(al.is_terminal());
    }

    #[test]
    fn history_tracking() {
        let mut al = default_loop();
        drive_happy_path(&mut al);
        assert_eq!(al.history().len(), 10);
        // First transition should be Idle → OrderReceived.
        assert!(matches!(al.history()[0].0, AutonomyState::Idle));
        assert!(matches!(al.history()[0].1, AutonomyEvent::OrderPlaced));
        assert_eq!(al.history()[0].2, 1);
    }

    #[test]
    fn error_event_always_transitions_to_failed() {
        let mut al = default_loop();
        al.process_event(AutonomyEvent::OrderPlaced, 1).unwrap();
        al.process_event(AutonomyEvent::DesignSelected, 2).unwrap();
        al.process_event(AutonomyEvent::Error("machine fire".into()), 3)
            .unwrap();
        assert!(matches!(al.state(), AutonomyState::Failed { .. }));
        if let AutonomyState::Failed { reason, .. } = al.state() {
            assert_eq!(reason, "machine fire");
        }
    }

    #[test]
    fn cannot_transition_from_complete() {
        let mut al = default_loop();
        drive_happy_path(&mut al);
        let res = al.process_event(AutonomyEvent::OrderPlaced, 100);
        assert!(matches!(
            res,
            Err(AutonomyError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn order_id_accessor() {
        assert!(AutonomyState::Idle.order_id().is_none());
        let s = AutonomyState::Printing {
            order_id: "X".into(),
            job_id: "J".into(),
        };
        assert_eq!(s.order_id(), Some("X"));
    }
}
