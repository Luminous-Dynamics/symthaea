// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
pub use calibrator::{TheoryCalibrator, soft_min};
pub use detector::ConflictDetector;
pub use phi_integration::{PhiEffResult, compute_phi_eff, effective_phi, thresholds};
pub use types::{
    AnchorKind, ConflictKind, ConflictMatrix, ConflictScore, EpistemicAction, MultiTheoryMetrics,
    TheoryCalibration, TheoryCalibrations, TheoryId,
};
