//! Feedback Variable Integration — Consensus-based proposal system.
//!
//! All feedback variable mutations (`prediction_confidence`, `fep_lr_boost`,
//! `exploration_urge`, `adaptive_threshold_scale`) are routed through attributed
//! proposals. Each cycle, proposals are collected via helper methods and
//! integrated using consensus (averaged adds, geometric mean scales) to produce
//! noise-resistant values for the next cycle.

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
    #[allow(dead_code)]
    pub fn proposals(&self) -> &[AttributedProposal] {
        &self.proposals
    }

    /// Dump all proposals as `(source, description)` pairs for trace logging.
    ///
    /// Each entry describes the proposal kind and magnitude, e.g.
    /// `("binding_strong", "Add(+0.0300)")`.
    pub fn dump_proposals(&self) -> Vec<(&'static str, String)> {
        self.proposals
            .iter()
            .map(|ap| {
                let desc = match ap.proposal {
                    FeedbackProposal::Add(d) => format!("Add({d:+.4})"),
                    FeedbackProposal::Scale(f) => format!("Scale({f:.4})"),
                    FeedbackProposal::Set(v) => format!("Set({v:.4})"),
                };
                (ap.source, desc)
            })
            .collect()
    }

    /// Integrate all proposals using consensus mode.
    ///
    /// Consensus integration strategy:
    /// - `Set` proposals: last one wins (they're rare and intentional)
    /// - `Add` proposals: averaged (consensus), then applied
    /// - `Scale` proposals: geometric mean, then applied
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

        // Average additive proposals (noise-resistant consensus)
        if !adds.is_empty() {
            let avg_delta: f64 = adds.iter().map(|(_, d)| d).sum::<f64>() / adds.len() as f64;
            value += avg_delta;
        }
        // Geometric mean of multiplicative proposals
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

/// Consensus-integrated feedback values computed at cycle end.
///
/// These noise-resistant values (averaged adds, geometric mean scales)
/// are stored for deferred writeback at the start of the next cycle.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ConsensusResult {
    /// Consensus-integrated prediction_confidence for next cycle
    pub consensus_confidence: f64,
    /// Consensus-integrated fep_lr_boost value for next cycle
    pub consensus_lr: f64,
    /// Consensus-integrated exploration_urge for next cycle
    pub consensus_exploration: f64,
    /// Consensus-integrated adaptive_threshold_scale for next cycle
    pub consensus_threshold: f64,
}

/// Decoupled feedback state for the cognitive loop.
///
/// All feedback variable mutations are routed through attributed proposals.
/// At cycle end, proposals are integrated using consensus (averaged adds,
/// geometric mean scales) and the result is stored for deferred writeback
/// at the next cycle start.
#[derive(Debug, Clone)]
pub(crate) struct FeedbackState {
    /// Proposals for `prediction_confidence` (0.0–1.0)
    pub confidence: ProposalCollector,
    /// Proposals for `fep_lr_boost` (1.0–3.0)
    pub learning_rate: ProposalCollector,
    /// Proposals for `exploration_urge` (0.0–1.0)
    pub exploration: ProposalCollector,
    /// Proposals for `adaptive_threshold_scale` (0.5–2.0)
    pub threshold: ProposalCollector,

    // ── Cycle-start values for consensus computation ────────────────────
    cycle_start_confidence: f64,
    cycle_start_lr: f64,
    cycle_start_exploration: f64,
    cycle_start_threshold: f64,

    // ── Cycle-end integration results (for telemetry / debugging) ───────
    /// Last integrated confidence result
    pub last_confidence_integration: Option<IntegrationResult>,
    /// Last integrated LR result
    pub last_lr_integration: Option<IntegrationResult>,
    /// Last integrated exploration result
    pub last_exploration_integration: Option<IntegrationResult>,
    /// Last integrated threshold result
    pub last_threshold_integration: Option<IntegrationResult>,

    /// Last computed consensus result (set by `end_cycle()`).
    pub last_consensus: Option<ConsensusResult>,

    // ── Deferred consensus writeback ────────────────────────────────────
    /// Consensus confidence to apply via helper at the start of the next cycle.
    /// Stored by `store_consensus_for_next_cycle()`, consumed by `apply_pending_consensus()`.
    pending_consensus_confidence: Option<f64>,
    /// Consensus LR to apply via helper at the start of the next cycle.
    pending_consensus_lr: Option<f64>,
    /// Consensus exploration to apply via helper at the start of the next cycle.
    pending_consensus_exploration: Option<f64>,
    /// Consensus threshold to apply via helper at the start of the next cycle.
    pending_consensus_threshold: Option<f64>,
}

impl FeedbackState {
    pub fn new() -> Self {
        Self {
            confidence: ProposalCollector::new(),
            learning_rate: ProposalCollector::new(),
            exploration: ProposalCollector::new(),
            threshold: ProposalCollector::new(),
            cycle_start_confidence: 0.5,
            cycle_start_lr: 1.0,
            cycle_start_exploration: 0.3,
            cycle_start_threshold: 1.0,
            last_confidence_integration: None,
            last_lr_integration: None,
            last_exploration_integration: None,
            last_threshold_integration: None,
            last_consensus: None,
            pending_consensus_confidence: None,
            pending_consensus_lr: None,
            pending_consensus_exploration: None,
            pending_consensus_threshold: None,
        }
    }

    /// Clear all collectors at cycle start.
    pub fn begin_cycle(&mut self) {
        self.confidence.clear();
        self.learning_rate.clear();
        self.exploration.clear();
        self.threshold.clear();
    }

    /// Integrate all feedback variables using consensus. Call at cycle end.
    ///
    /// Returns consensus-integrated values (averaged adds, geometric mean scales)
    /// for all 4 feedback variables.
    pub fn end_cycle(
        &mut self,
        _current_confidence: f64,
        _current_lr: f64,
        _current_exploration: f64,
        _current_threshold: f64,
    ) -> ConsensusResult {
        let base_confidence = self.cycle_start_confidence;
        let base_lr = self.cycle_start_lr;
        let base_exploration = self.cycle_start_exploration;
        let base_threshold = self.cycle_start_threshold;

        let conf_result = self.confidence.integrate(base_confidence, 0.0, 1.0);
        let lr_result = self.learning_rate.integrate(base_lr, 1.0, 3.0);
        let explore_result = self.exploration.integrate(base_exploration, 0.0, 1.0);
        let thresh_result = self.threshold.integrate(base_threshold, 0.5, 2.0);

        let consensus = ConsensusResult {
            consensus_confidence: conf_result.effective,
            consensus_lr: lr_result.effective,
            consensus_exploration: explore_result.effective,
            consensus_threshold: thresh_result.effective,
        };

        self.last_confidence_integration = Some(conf_result);
        self.last_lr_integration = Some(lr_result);
        self.last_exploration_integration = Some(explore_result);
        self.last_threshold_integration = Some(thresh_result);
        self.last_consensus = Some(consensus);

        consensus
    }

    /// Record cycle-start values for consensus computation.
    ///
    /// Call at cycle start (after `begin_cycle()`) with the current field values.
    pub fn snapshot_cycle_start(
        &mut self,
        confidence: f64,
        lr: f64,
        exploration: f64,
        threshold: f64,
    ) {
        self.cycle_start_confidence = confidence;
        self.cycle_start_lr = lr;
        self.cycle_start_exploration = exploration;
        self.cycle_start_threshold = threshold;
    }

    /// Store consensus-smoothed values for deferred application at the next cycle start.
    ///
    /// Called at cycle end. The values are applied via helpers by
    /// `apply_pending_consensus()` at the next cycle start.
    pub fn store_consensus_for_next_cycle(&mut self, consensus: &ConsensusResult) {
        self.pending_consensus_confidence = Some(consensus.consensus_confidence);
        self.pending_consensus_lr = Some(consensus.consensus_lr);
        self.pending_consensus_exploration = Some(consensus.consensus_exploration);
        self.pending_consensus_threshold = Some(consensus.consensus_threshold);
    }

    /// Return any pending consensus values for the caller to apply via helpers.
    ///
    /// Call at cycle start (after `begin_cycle()` + `snapshot_cycle_start()`).
    /// Returns `(confidence, lr, exploration, threshold)` — the caller routes
    /// these through `set_confidence` / `set_lr` / `set_exploration` / `set_threshold`
    /// helpers, which emit the Set proposals themselves.
    pub fn apply_pending_consensus(
        &mut self,
    ) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
        let confidence = self.pending_consensus_confidence.take();
        let lr = self.pending_consensus_lr.take();
        let exploration = self.pending_consensus_exploration.take();
        let threshold = self.pending_consensus_threshold.take();
        (confidence, lr, exploration, threshold)
    }

    /// How many total proposals were recorded this cycle.
    #[allow(dead_code)]
    pub fn total_proposals(&self) -> usize {
        self.confidence.len()
            + self.learning_rate.len()
            + self.exploration.len()
            + self.threshold.len()
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
        state
            .confidence
            .propose("subsys_a", FeedbackProposal::Add(0.05));
        state
            .confidence
            .propose("subsys_b", FeedbackProposal::Scale(0.98));
        state
            .learning_rate
            .propose("fep", FeedbackProposal::Add(0.1));
        assert_eq!(state.total_proposals(), 3);

        // End cycle with current values
        state.end_cycle(0.6, 1.2, 0.4, 1.0);
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

    #[test]
    fn test_dump_proposals_formats_correctly() {
        let mut collector = ProposalCollector::new();
        collector.propose("binding_strong", FeedbackProposal::Add(0.03));
        collector.propose("moral_harm", FeedbackProposal::Scale(0.85));
        collector.propose("init_reset", FeedbackProposal::Set(0.5));

        let dump = collector.dump_proposals();
        assert_eq!(dump.len(), 3);
        assert_eq!(dump[0].0, "binding_strong");
        assert!(dump[0].1.starts_with("Add("));
        assert_eq!(dump[1].0, "moral_harm");
        assert!(dump[1].1.starts_with("Scale("));
        assert_eq!(dump[2].0, "init_reset");
        assert!(dump[2].1.starts_with("Set("));
    }

    #[test]
    fn test_dump_proposals_empty_collector() {
        let collector = ProposalCollector::new();
        let dump = collector.dump_proposals();
        assert!(dump.is_empty());
    }

    /// Verify that consensus writebacks routed through `store_consensus_for_next_cycle`
    /// + `apply_pending_consensus` produce values across two cycles.
    #[test]
    fn test_consensus_writeback_routed_through_proposals() {
        let mut state = FeedbackState::new();

        // Cycle 1: no pending consensus yet
        state.begin_cycle();
        let (conf, lr, explore, thresh) = state.apply_pending_consensus();
        assert!(conf.is_none()); // No pending on first cycle
        assert!(lr.is_none());
        assert!(explore.is_none());
        assert!(thresh.is_none());

        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);
        state
            .confidence
            .propose("test", FeedbackProposal::Add(0.05));
        state
            .learning_rate
            .propose("test", FeedbackProposal::Scale(1.1));
        state
            .exploration
            .propose("test", FeedbackProposal::Add(0.02));
        state
            .threshold
            .propose("test", FeedbackProposal::Scale(0.95));

        let consensus = state.end_cycle(0.55, 1.1, 0.32, 0.95);

        // Store consensus for next cycle (simulating what cycle_phase_output does)
        state.store_consensus_for_next_cycle(&consensus);

        // Cycle 2: consensus should be applied
        state.begin_cycle();
        let (conf, lr, explore, thresh) = state.apply_pending_consensus();
        // Consensus values should be pending
        assert!(conf.is_some(), "consensus confidence should be pending");
        assert!(lr.is_some(), "consensus lr should be pending");
        assert!(explore.is_some(), "consensus exploration should be pending");
        assert!(thresh.is_some(), "consensus threshold should be pending");

        // Values should be finite and within clamp ranges
        assert!(conf.unwrap().is_finite());
        assert!(lr.unwrap().is_finite());
        assert!(explore.unwrap().is_finite());
        assert!(thresh.unwrap().is_finite());
    }
}
