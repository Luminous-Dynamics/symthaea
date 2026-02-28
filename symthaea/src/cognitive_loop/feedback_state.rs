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

    /// Integrate all proposals using consensus mode (default).
    ///
    /// Consensus integration strategy:
    /// - `Set` proposals: last one wins (they're rare and intentional)
    /// - `Add` proposals: averaged (consensus), then applied
    /// - `Scale` proposals: geometric mean, then applied
    ///
    /// Currently used in tests; will become the sole integration path
    /// when dual-write bridge is removed (see module doc).
    #[allow(dead_code)]
    pub fn integrate(&self, base_value: f64, clamp_min: f64, clamp_max: f64) -> IntegrationResult {
        self.integrate_with_mode(base_value, clamp_min, clamp_max, IntegrationMode::Consensus)
    }

    /// Integrate all proposals using sequential mode.
    ///
    /// Sequential mode matches the behavior of applying each proposal
    /// in order (sum adds, product scales). Use this for cutover
    /// validation — the result should match the direct-mutation value.
    pub fn integrate_sequential(
        &self,
        base_value: f64,
        clamp_min: f64,
        clamp_max: f64,
    ) -> IntegrationResult {
        self.integrate_with_mode(base_value, clamp_min, clamp_max, IntegrationMode::Sequential)
    }

    fn integrate_with_mode(
        &self,
        base_value: f64,
        clamp_min: f64,
        clamp_max: f64,
        mode: IntegrationMode,
    ) -> IntegrationResult {
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

        match mode {
            IntegrationMode::Consensus => {
                // Average additive proposals (noise-resistant consensus)
                if !adds.is_empty() {
                    let avg_delta: f64 =
                        adds.iter().map(|(_, d)| d).sum::<f64>() / adds.len() as f64;
                    value += avg_delta;
                }
                // Geometric mean of multiplicative proposals
                if !scales.is_empty() {
                    let log_sum: f64 = scales.iter().map(|(_, f)| f.ln()).sum::<f64>();
                    let geo_mean = (log_sum / scales.len() as f64).exp();
                    value *= geo_mean;
                }
            }
            IntegrationMode::Sequential => {
                // Sum additive proposals (matches sequential += behavior)
                if !adds.is_empty() {
                    let total_delta: f64 = adds.iter().map(|(_, d)| d).sum::<f64>();
                    value += total_delta;
                }
                // Product of multiplicative proposals (matches sequential *= behavior)
                if !scales.is_empty() {
                    let product: f64 = scales.iter().map(|(_, f)| f).product();
                    value *= product;
                }
            }
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

/// Integration strategy for combining proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum IntegrationMode {
    /// Consensus: average adds, geometric mean scales. Noise-resistant.
    /// Use for final production mode (reduces compounding drift).
    Consensus,
    /// Sequential: sum adds, product scales. Matches sequential mutation behavior.
    /// Use for cutover validation (result should match direct-mutation value).
    Sequential,
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

/// Divergence between direct-mutation values and proposal-integrated values.
///
/// A divergence of 0.0 means the proposal system perfectly reproduces
/// the direct-mutation behavior. Non-zero divergence indicates mutations
/// that bypass the proposal system (not routed through helpers).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FeedbackDivergence {
    pub confidence: f64,
    pub learning_rate: f64,
    pub exploration: f64,
    pub threshold: f64,
}

/// Decoupled feedback state for the cognitive loop.
///
/// Replaces direct `self.prediction_confidence += X` and `self.fep_lr_boost *= X`
/// mutations with attributed, integrated proposals.
///
/// Round 7 extends this to `exploration_urge` (27 sites) and
/// `adaptive_threshold_scale` (8 sites).
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

    // ── Cycle-start values for divergence computation ────────────────────
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
        }
    }

    /// Clear all collectors at cycle start.
    pub fn begin_cycle(&mut self) {
        self.confidence.clear();
        self.learning_rate.clear();
        self.exploration.clear();
        self.threshold.clear();
    }

    /// Integrate all feedback variables. Call at cycle end (Phase D: DECAY).
    ///
    /// Current values are produced by direct mutations (the old path).
    /// Sequential integration results are stored for comparison — these
    /// should match the direct-mutation values if all mutations go through helpers.
    ///
    /// Returns divergence between direct mutation and sequential integration
    /// for each variable (0.0 = perfect match).
    pub fn end_cycle(
        &mut self,
        current_confidence: f64,
        current_lr: f64,
        current_exploration: f64,
        current_threshold: f64,
    ) -> FeedbackDivergence {
        // Use sequential integration (sum/product) — should match direct mutation exactly
        let base_confidence = self.cycle_start_confidence;
        let base_lr = self.cycle_start_lr;
        let base_exploration = self.cycle_start_exploration;
        let base_threshold = self.cycle_start_threshold;

        let conf_result = self
            .confidence
            .integrate_sequential(base_confidence, 0.0, 1.0);
        let lr_result = self.learning_rate.integrate_sequential(base_lr, 1.0, 3.0);
        let explore_result = self
            .exploration
            .integrate_sequential(base_exploration, 0.0, 1.0);
        let thresh_result = self
            .threshold
            .integrate_sequential(base_threshold, 0.5, 2.0);

        let divergence = FeedbackDivergence {
            confidence: (current_confidence - conf_result.effective).abs(),
            learning_rate: (current_lr - lr_result.effective).abs(),
            exploration: (current_exploration - explore_result.effective).abs(),
            threshold: (current_threshold - thresh_result.effective).abs(),
        };

        self.last_confidence_integration = Some(conf_result);
        self.last_lr_integration = Some(lr_result);
        self.last_exploration_integration = Some(explore_result);
        self.last_threshold_integration = Some(thresh_result);

        divergence
    }

    /// Record cycle-start values for divergence computation.
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

    #[test]
    fn test_sequential_integration_sums_adds() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Add(0.1));
        collector.propose("b", FeedbackProposal::Add(0.3));
        // Sequential: sum of +0.1 and +0.3 = +0.4, applied to base 0.5 → 0.9
        let result = collector.integrate_sequential(0.5, 0.0, 1.0);
        assert!((result.effective - 0.9).abs() < 1e-10);
        // Compare with consensus (average): +0.2 → 0.7
        let consensus = collector.integrate(0.5, 0.0, 1.0);
        assert!((consensus.effective - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_sequential_integration_products_scales() {
        let mut collector = ProposalCollector::new();
        collector.propose("a", FeedbackProposal::Scale(0.9));
        collector.propose("b", FeedbackProposal::Scale(0.9));
        // Sequential: product 0.9 * 0.9 = 0.81, applied to base 1.0 → 0.81
        let result = collector.integrate_sequential(1.0, 0.0, 2.0);
        assert!((result.effective - 0.81).abs() < 1e-10);
        // Compare with consensus (geomean): 0.9 → 0.9
        let consensus = collector.integrate(1.0, 0.0, 2.0);
        assert!((consensus.effective - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_divergence_zero_when_fully_mediated() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);

        // Simulate helpers: propose and apply
        state
            .confidence
            .propose("test", FeedbackProposal::Add(0.1));
        state
            .learning_rate
            .propose("test", FeedbackProposal::Scale(1.05));

        // end_cycle with values matching sequential integration
        // conf: 0.5 + 0.1 = 0.6, lr: 1.0 * 1.05 = 1.05
        let divergence = state.end_cycle(0.6, 1.05, 0.3, 1.0);
        assert!(divergence.confidence < 1e-10);
        assert!(divergence.learning_rate < 1e-10);
        assert!(divergence.exploration < 1e-10);
        assert!(divergence.threshold < 1e-10);
    }

    #[test]
    fn test_divergence_nonzero_with_bypass() {
        let mut state = FeedbackState::new();
        state.begin_cycle();
        state.snapshot_cycle_start(0.5, 1.0, 0.3, 1.0);

        // Only confidence is proposed
        state
            .confidence
            .propose("test", FeedbackProposal::Add(0.1));
        // But actual confidence was changed by both proposal AND bypass
        let divergence = state.end_cycle(0.7, 1.0, 0.3, 1.0);
        // Integrated: 0.5 + 0.1 = 0.6, actual: 0.7, divergence: 0.1
        assert!((divergence.confidence - 0.1).abs() < 1e-10);
    }

    /// Multi-cycle divergence diagnostic: runs 50 cognitive cycles and checks that
    /// lr_divergence (fep_lr_boost) and threshold_divergence are near-zero,
    /// confirming all mutations go through helpers (100% mediated).
    #[test]
    fn test_multicycle_lr_divergence_near_zero() {
        use super::super::CognitiveLoopConfig;
        use super::super::CognitiveLoopService;

        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let mut max_lr_div = 0.0f64;
        let mut max_thresh_div = 0.0f64;
        let mut total_lr_proposals = 0usize;
        let mut total_thresh_proposals = 0usize;

        for i in 0..50 {
            let input = format!("divergence diagnostic cycle {i}");
            let _ = service.cycle(&input);

            // Collect divergence from the last integration
            if let Some(ref lr_int) = service.feedback_state.last_lr_integration {
                total_lr_proposals += lr_int.n_adds + lr_int.n_scales + lr_int.n_sets;
            }
            if let Some(ref thresh_int) = service.feedback_state.last_threshold_integration {
                total_thresh_proposals += thresh_int.n_adds + thresh_int.n_scales + thresh_int.n_sets;
            }

            // Compute divergence by re-running end_cycle logic
            let lr_actual = service.fep_lr_boost as f64;
            let thresh_actual = service.carryover.learning.adaptive_threshold_scale as f64;

            // The last_*_integration holds the sequential integration result.
            // Compare with actual values.
            if let Some(ref lr_int) = service.feedback_state.last_lr_integration {
                let div = (lr_actual - lr_int.effective).abs();
                if div > max_lr_div {
                    max_lr_div = div;
                }
            }
            if let Some(ref thresh_int) = service.feedback_state.last_threshold_integration {
                let div = (thresh_actual - thresh_int.effective).abs();
                if div > max_thresh_div {
                    max_thresh_div = div;
                }
            }
        }

        // We expect proposals to be generated (sanity check)
        assert!(
            total_lr_proposals > 100,
            "Expected 100+ lr proposals over 50 cycles, got {total_lr_proposals}"
        );
        assert!(
            total_thresh_proposals > 10,
            "Expected 10+ threshold proposals over 50 cycles, got {total_thresh_proposals}"
        );

        // Sequential integration should match direct mutations for 100% mediated fields.
        // Allow small tolerance for f32→f64 rounding in the integration path.
        assert!(
            max_lr_div < 0.01,
            "fep_lr_boost max divergence {max_lr_div:.6} exceeds 1% — \
             some mutations bypass helpers"
        );
        assert!(
            max_thresh_div < 0.01,
            "adaptive_threshold_scale max divergence {max_thresh_div:.6} exceeds 1% — \
             some mutations bypass helpers"
        );
    }
}
