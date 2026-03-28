//! Epistemic Conflict Detection Subsystem
//!
//! Detects, classifies, and tracks conflicts between 6 consciousness
//! theories (IIT, GWT, AST, PP, RPT, 4E). Provides:
//!
//! - **15 pairwise conflict scores** with typed `ConflictKind`
//! - **Reliability R** = softmin(consensus, coverage)
//! - **Effective Φ** = Φ × R^γ (monotonically cautious, INV-1)
//! - **Calibration** with bounded updates (INV-9)
//! - **Ground-truth anchors** for epistemic actions (INV-10)

pub mod calibrator;
pub mod detector;
pub mod phi_integration;
pub mod types;

// Re-export key types for ergonomic use
pub use calibrator::{soft_min, TheoryCalibrator};
pub use detector::ConflictDetector;
pub use phi_integration::{compute_phi_eff, effective_phi, thresholds, PhiEffResult};
pub use types::{
    AnchorKind, ConflictKind, ConflictMatrix, ConflictScore, EpistemicAction, MultiTheoryMetrics,
    TheoryCalibration, TheoryCalibrations, TheoryId,
};
