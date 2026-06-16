// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epistemic Conflict Types
//!
//! Core types for the epistemic conflict detection subsystem.
//! Includes theory identifiers, conflict taxonomy, epistemic actions
//! with ground-truth anchors (INV-10), and conflict scoring.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Theory Metrics (standalone, feature-independent)
// ─────────────────────────────────────────────────────────────────────────────

/// Six-theory consciousness metrics used by the conflict detector.
///
/// This is a standalone definition that can be constructed from
/// `language::multi_theory_consciousness::MultiTheoryMetrics` when the
/// `full_language` feature is enabled. This avoids coupling features.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MultiTheoryMetrics {
    /// Integrated Information Theory (Φ)
    pub phi: f64,
    /// Global Workspace Theory (broadcast)
    pub gwt: f64,
    /// Attention Schema Theory
    pub ast: f64,
    /// Predictive Processing
    pub pp: f64,
    /// Recurrent Processing Theory
    pub rpt: f64,
    /// 4E Cognition (embodiment)
    pub embodiment: f64,
    /// Unified weighted metric
    pub unified: f64,
}

impl MultiTheoryMetrics {
    /// Extract values as a 6-element array in canonical order.
    pub fn values(&self) -> [f64; 6] {
        [
            self.phi,
            self.gwt,
            self.ast,
            self.pp,
            self.rpt,
            self.embodiment,
        ]
    }
}

#[cfg(feature = "full_language")]
impl From<&crate::language::multi_theory_consciousness::MultiTheoryMetrics> for MultiTheoryMetrics {
    fn from(m: &crate::language::multi_theory_consciousness::MultiTheoryMetrics) -> Self {
        Self {
            phi: m.phi,
            gwt: m.gwt,
            ast: m.ast,
            pp: m.pp,
            rpt: m.rpt,
            embodiment: m.embodiment,
            unified: m.unified,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Theory Identifiers
// ─────────────────────────────────────────────────────────────────────────────

/// The six consciousness theories tracked by the multi-theory framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TheoryId {
    /// Integrated Information Theory (Φ)
    IIT,
    /// Global Workspace Theory (broadcast)
    GWT,
    /// Attention Schema Theory (attention model)
    AST,
    /// Predictive Processing (prediction error minimization)
    PP,
    /// Recurrent Processing Theory (recurrence depth)
    RPT,
    /// 4E Cognition (embodied, embedded, enacted, extended)
    FourE,
}

impl TheoryId {
    /// All theory variants in canonical order.
    pub const ALL: [TheoryId; 6] = [
        TheoryId::IIT,
        TheoryId::GWT,
        TheoryId::AST,
        TheoryId::PP,
        TheoryId::RPT,
        TheoryId::FourE,
    ];

    /// Index into a 6-element array.
    pub fn index(self) -> usize {
        match self {
            TheoryId::IIT => 0,
            TheoryId::GWT => 1,
            TheoryId::AST => 2,
            TheoryId::PP => 3,
            TheoryId::RPT => 4,
            TheoryId::FourE => 5,
        }
    }

    /// Human-readable short name.
    pub fn short_name(self) -> &'static str {
        match self {
            TheoryId::IIT => "IIT",
            TheoryId::GWT => "GWT",
            TheoryId::AST => "AST",
            TheoryId::PP => "PP",
            TheoryId::RPT => "RPT",
            TheoryId::FourE => "4E",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict Taxonomy (6 kinds, 1 per theory weakness)
// ─────────────────────────────────────────────────────────────────────────────

/// Typed conflict kinds — one per theory weakness.
///
/// Each variant indicates which theory is weak, enabling targeted
/// epistemic remediation via `recommended_action()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConflictKind {
    /// IIT weak: information integration has collapsed.
    IntegrationCollapse,
    /// GWT weak: no broadcast to global workspace.
    NoBroadcast,
    /// AST weak: attention is unstable or unfocused.
    AttentionalInstability,
    /// PP weak: predictions are unreliable / high surprise.
    UnreliablePrediction,
    /// RPT weak: recurrence too shallow for the task.
    ShallowRecurrence,
    /// 4E weak: representations are ungrounded / disembodied.
    UngroundedRepresentation,
}

impl ConflictKind {
    /// Which theory is weakened by this conflict.
    pub fn weak_theory(self) -> TheoryId {
        match self {
            ConflictKind::IntegrationCollapse => TheoryId::IIT,
            ConflictKind::NoBroadcast => TheoryId::GWT,
            ConflictKind::AttentionalInstability => TheoryId::AST,
            ConflictKind::UnreliablePrediction => TheoryId::PP,
            ConflictKind::ShallowRecurrence => TheoryId::RPT,
            ConflictKind::UngroundedRepresentation => TheoryId::FourE,
        }
    }

    /// Recommended epistemic action to resolve this conflict kind.
    pub fn recommended_action(self) -> EpistemicAction {
        match self {
            ConflictKind::IntegrationCollapse => EpistemicAction::Summarize,
            ConflictKind::NoBroadcast => EpistemicAction::Verify(AnchorKind::ReadOnlyQuery),
            ConflictKind::AttentionalInstability => EpistemicAction::Ask,
            ConflictKind::UnreliablePrediction => {
                EpistemicAction::Sense(AnchorKind::SensorMeasurement)
            }
            ConflictKind::ShallowRecurrence => EpistemicAction::Simulate,
            ConflictKind::UngroundedRepresentation => {
                EpistemicAction::Verify(AnchorKind::DeterministicProbe)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ground-Truth Anchors (INV-10)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolution authorities for ground-truth checks.
///
/// INV-10 requires that `Verify` and `Sense` actions always specify
/// an AnchorKind. This is enforced at the type level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnchorKind {
    /// Read-only system query (e.g., nix search, file read).
    ReadOnlyQuery,
    /// SHA-256 hash of file content.
    FileHash,
    /// Command with deterministic output.
    DeterministicProbe,
    /// System measurement (e.g., uptime, disk usage).
    SensorMeasurement,
}

impl AnchorKind {
    /// Human-readable description of the anchor.
    pub fn description(self) -> &'static str {
        match self {
            AnchorKind::ReadOnlyQuery => "read-only system query",
            AnchorKind::FileHash => "file content hash (SHA-256)",
            AnchorKind::DeterministicProbe => "deterministic probe command",
            AnchorKind::SensorMeasurement => "system sensor measurement",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Epistemic Actions
// ─────────────────────────────────────────────────────────────────────────────

/// Epistemic actions with ground-truth anchors (INV-10).
///
/// `Verify` and `Sense` require an `AnchorKind` — this is a type-level
/// guarantee that these actions always specify a resolution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicAction {
    /// Ask the human for clarification (human is the authority).
    Ask,
    /// Sense the environment via a ground-truth anchor.
    Sense(AnchorKind),
    /// Run an internal simulation (no external anchor needed).
    Simulate,
    /// Defer decision to a later cycle.
    Defer,
    /// Summarize and integrate current information.
    Summarize,
    /// Verify via a ground-truth anchor (INV-10: requires AnchorKind).
    Verify(AnchorKind),
}

impl EpistemicAction {
    /// Whether this action involves external observation.
    pub fn is_external(self) -> bool {
        matches!(
            self,
            EpistemicAction::Ask | EpistemicAction::Sense(_) | EpistemicAction::Verify(_)
        )
    }

    /// Whether this action is purely internal.
    pub fn is_internal(self) -> bool {
        matches!(
            self,
            EpistemicAction::Simulate | EpistemicAction::Summarize | EpistemicAction::Defer
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict Score
// ─────────────────────────────────────────────────────────────────────────────

/// Enriched conflict score between a pair of theories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictScore {
    /// Theory pair: (higher-index, lower-index) in canonical order.
    pub theory_a: TheoryId,
    pub theory_b: TheoryId,

    /// Conflict magnitude in [0, 1].
    pub magnitude: f64,

    /// Typed conflict classification.
    pub kind: ConflictKind,

    /// Recommended epistemic action to resolve the conflict.
    pub recommended_action: EpistemicAction,

    /// Expected value of information for resolving this conflict.
    pub evoi: f64,

    /// Whether this conflict has persisted beyond the chronic threshold.
    pub is_chronic: bool,

    /// Rolling history of magnitudes (for trend detection).
    pub history: VecDeque<f64>,
}

impl ConflictScore {
    /// Create a new conflict score with default history.
    pub fn new(theory_a: TheoryId, theory_b: TheoryId, magnitude: f64, kind: ConflictKind) -> Self {
        Self {
            theory_a,
            theory_b,
            magnitude,
            kind,
            recommended_action: kind.recommended_action(),
            evoi: magnitude, // initial EVoI = magnitude (prior)
            is_chronic: false,
            history: VecDeque::with_capacity(32),
        }
    }

    /// Update magnitude and push to history.
    pub fn update(&mut self, new_magnitude: f64) {
        self.history.push_back(self.magnitude);
        if self.history.len() > 32 {
            self.history.pop_front();
        }
        self.magnitude = new_magnitude;
    }

    /// Trend: positive = worsening, negative = improving.
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let n = self.history.len();
        let recent = self.history.iter().skip(n / 2).sum::<f64>() / (n / 2) as f64;
        let older = self.history.iter().take(n / 2).sum::<f64>() / (n / 2) as f64;
        recent - older
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conflict Matrix (15 pairwise conflicts)
// ─────────────────────────────────────────────────────────────────────────────

/// 15 pairwise conflicts between 6 theories (C(6,2) = 15).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMatrix {
    /// All 15 pairwise conflict scores.
    pub conflicts: Vec<ConflictScore>,

    /// Total epistemic entropy: -Σ p_i ln(p_i) over normalized magnitudes.
    pub total_entropy: f64,

    /// The dominant (highest-magnitude) conflict, if any.
    pub dominant: Option<(TheoryId, TheoryId)>,

    /// Dominant conflict kind.
    pub dominant_kind: Option<ConflictKind>,
}

impl ConflictMatrix {
    /// Number of pairwise conflicts: C(6,2) = 15.
    pub const NUM_PAIRS: usize = 15;

    /// Create from a set of conflict scores.
    pub fn new(conflicts: Vec<ConflictScore>) -> Self {
        debug_assert_eq!(conflicts.len(), Self::NUM_PAIRS);

        let total_entropy = Self::compute_entropy(&conflicts);
        let dominant = conflicts
            .iter()
            .max_by(|a, b| a.magnitude.total_cmp(&b.magnitude))
            .map(|c| (c.theory_a, c.theory_b));
        let dominant_kind = conflicts
            .iter()
            .max_by(|a, b| a.magnitude.total_cmp(&b.magnitude))
            .map(|c| c.kind);

        Self {
            conflicts,
            total_entropy,
            dominant,
            dominant_kind,
        }
    }

    /// Compute epistemic entropy over conflict magnitudes.
    fn compute_entropy(conflicts: &[ConflictScore]) -> f64 {
        let total: f64 = conflicts.iter().map(|c| c.magnitude).sum();
        if total < 1e-10 {
            return 0.0;
        }
        conflicts
            .iter()
            .map(|c| {
                let p = c.magnitude / total;
                if p > 1e-10 { -p * p.ln() } else { 0.0 }
            })
            .sum()
    }

    /// Get conflict score between two specific theories.
    pub fn get(&self, a: TheoryId, b: TheoryId) -> Option<&ConflictScore> {
        self.conflicts
            .iter()
            .find(|c| (c.theory_a == a && c.theory_b == b) || (c.theory_a == b && c.theory_b == a))
    }

    /// Maximum conflict magnitude.
    pub fn max_magnitude(&self) -> f64 {
        self.conflicts
            .iter()
            .map(|c| c.magnitude)
            .fold(0.0_f64, f64::max)
    }

    /// Mean conflict magnitude.
    pub fn mean_magnitude(&self) -> f64 {
        if self.conflicts.is_empty() {
            return 0.0;
        }
        self.conflicts.iter().map(|c| c.magnitude).sum::<f64>() / self.conflicts.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Theory Calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Per-theory calibration state (Brier score tracking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryCalibration {
    /// Theory this calibration applies to.
    pub theory: TheoryId,

    /// Reliability weight in [0, 1], derived from calibration data.
    /// Bounded updates: Δw ≤ 0.05 per step (INV-9).
    pub reliability: f64,

    /// Running Brier score (lower = better calibrated).
    pub brier_score: f64,

    /// Number of calibration observations.
    pub observation_count: usize,

    /// Sum of squared errors for Brier computation.
    brier_sum: f64,
}

impl TheoryCalibration {
    /// Create with conservative defaults (cold start, INV-9 / FM-6).
    pub fn new(theory: TheoryId) -> Self {
        Self {
            theory,
            reliability: 0.5,  // conservative prior
            brier_score: 0.25, // uninformative prior (0.5^2)
            observation_count: 0,
            brier_sum: 0.0,
        }
    }

    /// Update with a new observation. Bounded by INV-9 (Δw ≤ 0.05).
    pub fn update(&mut self, predicted: f64, actual: f64) {
        const DELTA_W_MAX: f64 = 0.05;

        let error = (predicted - actual).powi(2);
        self.brier_sum += error;
        self.observation_count += 1;
        self.brier_score = self.brier_sum / self.observation_count as f64;

        // Derive new reliability from Brier score: R = 1 - brier
        let new_reliability = (1.0 - self.brier_score).clamp(0.0, 1.0);

        // Apply bounded update (INV-9)
        let delta = (new_reliability - self.reliability).clamp(-DELTA_W_MAX, DELTA_W_MAX);
        self.reliability = (self.reliability + delta).clamp(0.0, 1.0);
    }

    /// Whether we have enough data for confident calibration.
    pub fn is_cold(&self) -> bool {
        self.observation_count < 10
    }
}

/// Collection of calibrations for all 6 theories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TheoryCalibrations {
    calibrations: [TheoryCalibration; 6],
}

impl TheoryCalibrations {
    /// Create with conservative defaults for all theories.
    pub fn new() -> Self {
        Self {
            calibrations: [
                TheoryCalibration::new(TheoryId::IIT),
                TheoryCalibration::new(TheoryId::GWT),
                TheoryCalibration::new(TheoryId::AST),
                TheoryCalibration::new(TheoryId::PP),
                TheoryCalibration::new(TheoryId::RPT),
                TheoryCalibration::new(TheoryId::FourE),
            ],
        }
    }

    /// Get calibration for a specific theory.
    pub fn get(&self, theory: TheoryId) -> &TheoryCalibration {
        &self.calibrations[theory.index()]
    }

    /// Get mutable calibration for a specific theory.
    pub fn get_mut(&mut self, theory: TheoryId) -> &mut TheoryCalibration {
        &mut self.calibrations[theory.index()]
    }

    /// Whether any theory is cold (insufficient calibration data).
    pub fn any_cold(&self) -> bool {
        self.calibrations.iter().any(|c| c.is_cold())
    }

    /// Get all reliabilities as an array.
    pub fn reliabilities(&self) -> [f64; 6] {
        [
            self.calibrations[0].reliability,
            self.calibrations[1].reliability,
            self.calibrations[2].reliability,
            self.calibrations[3].reliability,
            self.calibrations[4].reliability,
            self.calibrations[5].reliability,
        ]
    }
}

impl Default for TheoryCalibrations {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theory_id_index_roundtrip() {
        for t in TheoryId::ALL {
            assert_eq!(TheoryId::ALL[t.index()], t);
        }
    }

    #[test]
    fn test_conflict_kind_recommended_action_requires_anchor() {
        // INV-10: Verify and Sense always have AnchorKind
        for kind in [
            ConflictKind::IntegrationCollapse,
            ConflictKind::NoBroadcast,
            ConflictKind::AttentionalInstability,
            ConflictKind::UnreliablePrediction,
            ConflictKind::ShallowRecurrence,
            ConflictKind::UngroundedRepresentation,
        ] {
            let action = kind.recommended_action();
            match action {
                EpistemicAction::Verify(anchor) => {
                    // AnchorKind is present — INV-10 satisfied
                    let _ = anchor.description();
                }
                EpistemicAction::Sense(anchor) => {
                    let _ = anchor.description();
                }
                _ => {} // Ask, Simulate, Defer, Summarize don't need anchors
            }
        }
        // All conflict kinds produce a valid recommended action (is_external or is_internal)
        let action = ConflictKind::IntegrationCollapse.recommended_action();
        assert!(
            action.is_external() || action.is_internal(),
            "recommended_action should be either external or internal"
        );
    }

    #[test]
    fn test_calibration_bounded_update_inv9() {
        let mut cal = TheoryCalibration::new(TheoryId::IIT);
        let initial = cal.reliability;

        // Even with perfect predictions, single-step change ≤ 0.05
        cal.update(0.0, 0.0); // perfect prediction
        assert!(
            (cal.reliability - initial).abs() <= 0.05 + 1e-10,
            "INV-9 violated: Δw = {}",
            (cal.reliability - initial).abs()
        );
    }

    #[test]
    fn test_conflict_matrix_entropy_bounds() {
        let conflicts: Vec<ConflictScore> = (0..15)
            .map(|i| {
                let a = TheoryId::ALL[i % 6];
                let b = TheoryId::ALL[(i + 1) % 6];
                ConflictScore::new(a, b, 0.5, ConflictKind::IntegrationCollapse)
            })
            .collect();
        let matrix = ConflictMatrix::new(conflicts);
        // Entropy should be non-negative
        assert!(matrix.total_entropy >= 0.0);
    }

    #[test]
    fn test_conflict_score_trend() {
        let mut score = ConflictScore::new(
            TheoryId::IIT,
            TheoryId::GWT,
            0.3,
            ConflictKind::IntegrationCollapse,
        );
        // Push increasing values — trend should be positive (worsening)
        for i in 0..10 {
            score.update(0.3 + i as f64 * 0.05);
        }
        assert!(
            score.trend() > 0.0,
            "Trend should be positive for worsening conflict"
        );
    }
}
