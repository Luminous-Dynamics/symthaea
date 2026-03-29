// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration constants for the Federated Learning Coordinator Zome.

// =============================================================================
// SECURITY: Authorization & Rate Limiting (F-03, F-06)
// =============================================================================

// MAX_NODE_ID_LENGTH is defined in federated_learning_integrity (canonical source)

/// Maximum submissions per agent per minute (F-06)
pub(crate) const MAX_SUBMISSIONS_PER_MINUTE: u32 = 60;
/// Coordinator role identifier
pub(crate) const COORDINATOR_ROLE: &str = "fl_coordinator";
/// Byzantine detector role identifier
pub(crate) const DETECTOR_ROLE: &str = "byzantine_detector";

// =============================================================================
// BYZANTINE DETECTION CONFIGURATION
// =============================================================================

/// Configurable Byzantine detection confidence threshold.
/// Gradients with detection confidence >= this value are REJECTED.
/// Range: 0.0 to 1.0, higher = stricter (fewer false positives, more false negatives)
pub(crate) const BYZANTINE_REJECTION_THRESHOLD: f32 = 0.7;

/// Minimum trust score for gradient acceptance.
/// Gradients from nodes with trust_score < this are flagged for detection.
pub(crate) const MIN_TRUST_SCORE_THRESHOLD: f64 = 0.3;

/// Number of hierarchy levels for HierarchicalDetector
pub(crate) const DETECTION_HIERARCHY_LEVELS: usize = 3;

/// Minimum cluster size for hierarchical detection
pub(crate) const DETECTION_MIN_CLUSTER_SIZE: usize = 2;

// =============================================================================
// SEC-002 FIX: Secure Coordinator Bootstrap Constants
// =============================================================================

/// Default bootstrap window duration in seconds (24 hours)
pub(crate) const DEFAULT_BOOTSTRAP_WINDOW_SECONDS: i64 = 86400;
/// Minimum guardians required for coordinator election (3 of 5 by default)
pub(crate) const DEFAULT_MIN_GUARDIANS: u32 = 3;
/// Default minimum votes for coordinator approval
pub(crate) const DEFAULT_MIN_VOTES: u32 = 3;
/// Maximum guardians allowed
pub(crate) const DEFAULT_MAX_GUARDIANS: u32 = 5;

// =============================================================================
// Coordinator Term Rotation
// =============================================================================

/// Default coordinator term duration in seconds (7 days)
pub(crate) const DEFAULT_TERM_DURATION_SECONDS: i64 = 604800;
/// Election window for re-election after term expiry (24 hours)
pub(crate) const DEFAULT_ROTATION_ELECTION_WINDOW_SECONDS: i64 = 86400;
/// Grace period after term expiry before credential becomes invalid (1 hour)
/// This prevents a gap between term expiry and new coordinator election
pub(crate) const TERM_GRACE_PERIOD_SECONDS: i64 = 3600;

// =============================================================================
// REPUTATION DECAY & QUARANTINE
// =============================================================================

/// Reputation decay factor per round-equivalent interval.
/// Applied lazily on retrieval: R_decayed = R_floor + (R - R_floor) * DECAY^elapsed_intervals
/// Range: 0.5..1.0 (0.95 = ~5% decay per interval, slow convergence toward floor)
pub(crate) const REPUTATION_DECAY_FACTOR: f64 = 0.95;

/// Duration of one decay interval in seconds (1 day = 86400).
/// Decay accumulates proportionally: 7 days idle = 7 intervals of decay.
pub(crate) const REPUTATION_DECAY_INTERVAL_SECONDS: i64 = 86400;

/// Minimum reputation floor. Decay never reduces reputation below this value.
/// Nodes at the floor are quarantined (cannot submit gradients).
pub(crate) const REPUTATION_FLOOR: f32 = 0.1;

/// Minimum reputation required to submit gradients.
/// Nodes below this threshold are quarantined until their reputation recovers.
pub(crate) const MIN_REPUTATION_FOR_SUBMISSION: f32 = 0.15;
