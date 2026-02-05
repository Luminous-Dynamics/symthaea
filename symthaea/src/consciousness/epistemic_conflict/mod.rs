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

pub mod types;
pub mod detector;
pub mod calibrator;
pub mod phi_integration;

// Re-export key types for ergonomic use
pub use types::{
    TheoryId, ConflictKind, AnchorKind, EpistemicAction,
    ConflictScore, ConflictMatrix, MultiTheoryMetrics,
    TheoryCalibration, TheoryCalibrations,
};
pub use detector::ConflictDetector;
pub use calibrator::{TheoryCalibrator, soft_min};
pub use phi_integration::{effective_phi, compute_phi_eff, PhiEffResult, thresholds};
