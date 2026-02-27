//! Feedback Variable Decoupling — Phase 2.2 of the Great Refactor
//!
//! Replaces direct `self.prediction_confidence += X` mutations with attributed
//! proposals that are collected and integrated once per cycle.
//!
//! # Current mode: Dual-write bridge
//!
//! Helper methods on `CognitiveLoopService` (in `helpers/feedback_helpers.rs`)
//! both record proposals AND apply the direct mutation, preserving exact behavior.
//! The `integrate()` step computes what the proposal-based value *would* be,
//! enabling verification before the full swap.
//!
//! # Future mode: Proposal-only integration
//!
//! Remove direct mutations from helpers; `integrate()` becomes the sole authority.

use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════════
// PROPOSAL TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// A single proposed change to a feedback variable.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FeedbackProposal {
    /// Additive delta: `value += delta`
    Add(f64),
    /// Multiplicative factor: `value *= factor`
    Scale(f64),
    /// Hard reset to a specific value (used sparingly: inference mode init, etc.)
    Set(f64),
}

/// An attributed proposal — who proposed what.
#[derive(Debug, Clone)]
pub(crate) struct AttributedProposal {
    /// Subsystem that made this proposal (static str for zero-alloc attribution)
    pub source: &'static str,
    /// The proposal itself
    pub proposal: FeedbackProposal,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROPOSAL COLLECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Collects proposals from subsystems during a cycle, then integrates them.
///
/// Reset at the start of each cycle via `clear()`.
#[derive(Debug, Clone)]
pub(crate) struct ProposalCollector {
    proposals: Vec<AttributedProposal>,
}

impl ProposalCollector {
    pub fn new() -> Self {
        Self {
            proposals: Vec::with_capacity(32),
        }
    }

    /// Record a proposal from a named subsystem.
    pub fn propose(&mut self, source: &'static str, proposal: FeedbackProposal) {
        self.proposals.push(AttributedProposal { source, proposal });
    }

    /// Number of proposals collected this cycle.
    pub fn len(&self) -> usize {
        self.proposals.len()
    }

    /// Clear all proposals (called at cycle start).
    pub fn clear(&mut self) {
        self.proposals.clear();
    }

    /// Iterate over proposals for inspection.
    pub fn proposals(&self) -> &[AttributedProposal] {
        &self.proposals
    }

    /// Integrate all proposals into a single effective value.
    ///
    /// Integration strategy:
    /// - `Set` proposals: last one wins (they're rare and intentional)
    /// - `Add` proposals: averaged (consensus), then applied
    /// - `Scale` proposals: geometric mean, then applied
    ///
    /// Returns `(effective_value, attribution_summary)`.
    pub fn integrate(&self, base_value: f64, clamp_min: f64, clamp_max: f64) -> IntegrationResult {
        let mut sets: Vec<(&'static str, f64)> = Vec::new();
        let mut adds: Vec<(&'static str, f64)> = Vec::new();
        let mut scales: Vec<(&'static str, f64)> = Vec::new();

        for ap in &self.proposals {
            match ap.proposal {
                FeedbackProposal::Set(v) => sets.push((ap.source, v)),
                FeedbackProposal::Add(d) => adds.push((ap.source, d)),
                FeedbackProposal::Scale(f) => scales.push((ap.source, f)),
            }
        }

        // Start from base or last Set
        let mut value = if let Some((_, v)) = sets.last() {
            *v
        } else {
            base_value
        };

        // Apply averaged additive proposals
        if !adds.is_empty() {
            let avg_delta: f64 = adds.iter().map(|(_, d)| d).sum::<f64>() / adds.len() as f64;
            value += avg_delta;
        }

        // Apply geometric mean of multiplicative proposals
        if !scales.is_empty() {
            let log_sum: f64 = scales.iter().map(|(_, f)| f.ln()).sum::<f64>();
            let geo_mean = (log_sum / scales.len() as f64).exp();
            value *= geo_mean;
        }

        value = value.clamp(clamp_min, clamp_max);

        IntegrationResult {
            effective: value,
            n_sets: sets.len(),
            n_adds: adds.len(),
            n_scales: scales.len(),
        }
    }
}

impl Default for ProposalCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of integrating proposals.
#[derive(Debug, Clone)]
pub(crate) struct IntegrationResult {
    pub effective: f64,
    pub n_sets: usize,
    pub n_adds: usize,
    pub n_scales: usize,
}

impl fmt::Display for IntegrationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "eff={:.4} ({}S+{}A+{}M)",
            self.effective, self.n_sets, self.n_adds, self.n_scales
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK STATE — top-level container
// ═══════════════════════════════════════════════════════════════════════════════

/// Decoupled feedback state for the cognitive loop.
///
/// Replaces the 68-site `prediction_confidence` and 31-site `fep_lr_boost`
/// direct mutations with attributed, integrated proposals.
#[derive(Debug, Clone)]
pub(crate) struct FeedbackState {
    /// Proposals for `prediction_confidence` (0.0–1.0)
    pub confidence: ProposalCollector,
    /// Proposals for `fep_lr_boost` (1.0–3.0)
    pub learning_rate: ProposalCollector,

    // ── Cycle-end integration results (for telemetry / debugging) ───────
    /// Last integrated confidence result
    pub last_confidence_integration: Option<IntegrationResult>,
    /// Last integrated LR result
    pub last_lr_integration: Option<IntegrationResult>,
}

impl FeedbackState {
    pub fn new() -> Self {
        Self {
            confidence: ProposalCollector::new(),
            learning_rate: ProposalCollector::new(),
            last_confidence_integration: None,
            last_lr_integration: None,
        }
    }

    /// Clear both collectors at cycle start.
    pub fn begin_cycle(&mut self) {
        self.confidence.clear();
        self.learning_rate.clear();
    }

    /// Integrate both feedback variables. Call at cycle end (Phase D: DECAY).
    ///
    /// `current_confidence` and `current_lr` are the values produced by direct
    /// mutations (the old path). The integration results are stored for
    /// comparison and telemetry.
    pub fn end_cycle(&mut self, current_confidence: f64, current_lr: f64) {
        self.last_confidence_integration =
            Some(self.confidence.integrate(current_confidence, 0.0, 1.0));
        self.last_lr_integration =
            Some(self.learning_rate.integrate(current_lr, 1.0, 3.0));
    }

    /// How many total proposals were recorded this cycle.
    pub fn total_proposals(&self) -> usize {
        self.confidence.len() + self.learning_rate.len()
    }
}

impl Default for FeedbackState {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_collector_returns_base() {
        let collector = ProposalCollector::new();
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.5).abs() < 1e-10);
        assert_eq!(result.n_sets, 0);
        assert_eq!(result.n_adds, 0);
        assert_eq!(result.n_scales, 0);
    }

    #[test]
    fn test_additive_proposals_averaged() {
        let mut collector = ProposalCollector::new();
        collector.propose("subsystem_a", FeedbackProposal::Add(0.1));
        collector.propose("subsystem_b", FeedbackProposal::Add(0.3));
        // Average of +0.1 and +0.3 = +0.2, applied to base 0.5 → 0.7
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.7).abs() < 1e-10);
        assert_eq!(result.n_adds, 2);
    }

    #[test]
    fn test_multiplicative_proposals_geometric_mean() {
        let mut collector = ProposalCollector::new();
        // Two scale factors: 0.9 and 1.1 → geo mean = sqrt(0.99) ≈ 0.995
        collector.propose("subsystem_a", FeedbackProposal::Scale(0.9));
        collector.propose("subsystem_b", FeedbackProposal::Scale(1.1));
        let result = collector.integrate(1.0, 0.0, 2.0);
        let expected = (0.9f64 * 1.1).sqrt(); // sqrt(0.99)
        assert!((result.effective - expected).abs() < 1e-10);
        assert_eq!(result.n_scales, 2);
    }

    #[test]
    fn test_set_overrides_base() {
        let mut collector = ProposalCollector::new();
        collector.propose("init", FeedbackProposal::Set(0.3));
        collector.propose("override", FeedbackProposal::Set(0.7));
        // Last Set wins
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 0.7).abs() < 1e-10);
        assert_eq!(result.n_sets, 2);
    }

    #[test]
    fn test_mixed_proposals() {
        let mut collector = ProposalCollector::new();
        collector.propose("base", FeedbackProposal::Set(0.5));
        collector.propose("boost_a", FeedbackProposal::Add(0.1));
        collector.propose("boost_b", FeedbackProposal::Add(0.1));
        collector.propose("scale_a", FeedbackProposal::Scale(1.2));
        // Set → 0.5, avg add → 0.1 → 0.6, geo_mean scale → 1.2 → 0.72
        let result = collector.integrate(0.0, 0.0, 1.0);
        assert!((result.effective - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_clamping_enforced() {
        let mut collector = ProposalCollector::new();
        collector.propose("huge", FeedbackProposal::Add(10.0));
        let result = collector.integrate(0.5, 0.0, 1.0);
        assert!((result.effective - 1.0).abs() < 1e-10);

        let mut collector2 = ProposalCollector::new();
        collector2.propose("negative", FeedbackProposal::Add(-10.0));
        let result2 = collector2.integrate(0.5, 0.0, 1.0);
        assert!((result2.effective - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_feedback_state_lifecycle() {
        let mut state = FeedbackState::new();

        // Begin cycle
        state.begin_cycle();
        assert_eq!(state.total_proposals(), 0);

        // Proposals during cycle
        state.confidence.propose("subsys_a", FeedbackProposal::Add(0.05));
        state.confidence.propose("subsys_b", FeedbackProposal::Scale(0.98));
        state.learning_rate.propose("fep", FeedbackProposal::Add(0.1));
        assert_eq!(state.total_proposals(), 3);

        // End cycle with current values
        state.end_cycle(0.6, 1.2);
        assert!(state.last_confidence_integration.is_some());
        assert!(state.last_lr_integration.is_some());

        // Next cycle clears
        state.begin_cycle();
        assert_eq!(state.total_proposals(), 0);
        // But last integration result persists
        assert!(state.last_confidence_integration.is_some());
    }

    #[test]
    fn test_lr_clamping_range() {
        let mut collector = ProposalCollector::new();
        collector.propose("huge", FeedbackProposal::Add(10.0));
        let result = collector.integrate(1.5, 1.0, 3.0);
        assert!((result.effective - 3.0).abs() < 1e-10);

        let mut collector2 = ProposalCollector::new();
        collector2.propose("negative", FeedbackProposal::Add(-10.0));
        let result2 = collector2.integrate(1.5, 1.0, 3.0);
        assert!((result2.effective - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_integration_result() {
        let result = IntegrationResult {
            effective: 0.7123,
            n_sets: 1,
            n_adds: 3,
            n_scales: 2,
        };
        let s = format!("{}", result);
        assert!(s.contains("0.7123"));
        assert!(s.contains("1S+3A+2M"));
    }

    #[test]
    fn test_proposal_collector_clear() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(1.0));
        collector.propose("b", FeedbackProposal::Add(2.0));
        assert_eq!(collector.len(), 2);
        collector.clear();
        assert_eq!(collector.len(), 0);
        assert!(collector.proposals().is_empty());
    }
}
