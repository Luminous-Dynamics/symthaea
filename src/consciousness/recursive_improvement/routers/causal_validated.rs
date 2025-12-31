//! # Causal Emergence-Validated Router
//!
//! Uses Causal Emergence (CE) to validate and adapt consciousness-guided routing.
//! When consciousness shows positive causal emergence over micro-level processing,
//! the router enables full conscious routing. Otherwise, it falls back to simpler strategies.
//!
//! ## Key Concepts
//!
//! - **Effective Information (EI)**: Measures causal power of a system
//! - **Causal Emergence**: CE = macro_EI - micro_EI (positive = consciousness has causal power)
//! - **Routing Modes**: Conscious (CE > threshold), Fallback (CE <= threshold), Calibrating
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::recursive_improvement::routers::*;
//!
//! let mut router = CausalValidatedRouter::new(CausalValidatedConfig::default());
//!
//! // Record state transitions for CE computation
//! router.record_transition(micro_state, macro_state, next_micro, next_macro);
//!
//! // Run cycles to update CE
//! router.cycle(0.001);
//!
//! // Get validated routing decision
//! let decision = router.route_validated(&current_state);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// Import from types module
use super::{
    RoutingStrategy, PhaseLockedPlan,
    LatentConsciousnessState,
};

// Import real OscillatoryRouter (type mismatch fixed December 30, 2025)
use super::{OscillatoryRouter, OscillatoryRouterConfig, OscillatoryRouterSummary};

// ═══════════════════════════════════════════════════════════════════════════
// OSCILLATORY ROUTER INTEGRATION (Phase 5G Complete - December 30, 2025)
// Now using real OscillatoryRouter with proper type definitions
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// EFFECTIVE INFORMATION
// ═══════════════════════════════════════════════════════════════════════════

/// Effective Information measurement
///
/// EI quantifies the causal power of a system by measuring how much
/// information the system's current state provides about its future state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveInformation {
    /// EI value in bits
    pub ei_bits: f64,
    /// Sample size used for estimation
    pub sample_size: usize,
    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
    /// Determinism component (how predictable are outputs)
    pub determinism: f64,
    /// Degeneracy component (how many inputs map to same output)
    pub degeneracy: f64,
}

impl EffectiveInformation {
    /// EI = Determinism - Degeneracy (approximate decomposition)
    pub fn compute(transitions: &[(Vec<f64>, Vec<f64>)]) -> Self {
        if transitions.is_empty() {
            return Self {
                ei_bits: 0.0,
                sample_size: 0,
                confidence_interval: (0.0, 0.0),
                determinism: 0.0,
                degeneracy: 0.0,
            };
        }

        let n = transitions.len();

        // Discretize states for entropy calculation
        let num_bins = (n as f64).sqrt().max(4.0) as usize;

        // Build transition counts
        let mut input_counts = HashMap::new();
        let mut output_counts = HashMap::new();
        let mut joint_counts = HashMap::new();

        for (input, output) in transitions {
            let in_bin = Self::discretize(input, num_bins);
            let out_bin = Self::discretize(output, num_bins);

            *input_counts.entry(in_bin.clone()).or_insert(0) += 1;
            *output_counts.entry(out_bin.clone()).or_insert(0) += 1;
            *joint_counts.entry((in_bin, out_bin)).or_insert(0) += 1;
        }

        // Compute entropies
        let h_input = Self::entropy_from_counts(&input_counts, n);
        let h_output = Self::entropy_from_counts(&output_counts, n);
        let h_joint = Self::joint_entropy_from_counts(&joint_counts, n);

        // Mutual information (approximates EI for max-entropy input)
        let mi = h_input + h_output - h_joint;

        // Determinism = H(X) - H(X|Y) ≈ MI when input is max-entropy
        let determinism = mi / h_input.max(0.001);

        // Degeneracy = how many inputs map to same output
        let degeneracy = 1.0 - (output_counts.len() as f64 / input_counts.len() as f64).min(1.0);

        // Confidence interval (rough approximation)
        let std_err = (mi / (n as f64).sqrt()).max(0.01);

        Self {
            ei_bits: mi.max(0.0),
            sample_size: n,
            confidence_interval: ((mi - 1.96 * std_err).max(0.0), mi + 1.96 * std_err),
            determinism: determinism.clamp(0.0, 1.0),
            degeneracy: degeneracy.clamp(0.0, 1.0),
        }
    }

    fn discretize(state: &[f64], num_bins: usize) -> Vec<usize> {
        state.iter()
            .map(|&v| ((v.clamp(0.0, 1.0) * (num_bins - 1) as f64) as usize).min(num_bins - 1))
            .collect()
    }

    fn entropy_from_counts(counts: &HashMap<Vec<usize>, usize>, total: usize) -> f64 {
        counts.values()
            .map(|&c| {
                let p = c as f64 / total as f64;
                if p > 1e-10 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }

    fn joint_entropy_from_counts(counts: &HashMap<(Vec<usize>, Vec<usize>), usize>, total: usize) -> f64 {
        counts.values()
            .map(|&c| {
                let p = c as f64 / total as f64;
                if p > 1e-10 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL EMERGENCE
// ═══════════════════════════════════════════════════════════════════════════

/// Causal Emergence measurement
///
/// CE = macro_EI - micro_EI
/// - Positive CE: Consciousness has causal power over micro-level
/// - Negative CE: Micro-level is more causally informative (reductive)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEmergence {
    /// Micro-level EI (neuron/primitive level)
    pub micro_ei: EffectiveInformation,
    /// Macro-level EI (consciousness level)
    pub macro_ei: EffectiveInformation,
    /// CE = macro_ei - micro_ei
    pub emergence: f64,
    /// Whether consciousness has causal power (CE > 0)
    pub causally_emergent: bool,
    /// Confidence in emergence (based on sample sizes)
    pub confidence: f64,
}

impl CausalEmergence {
    /// Compute causal emergence from micro and macro transitions
    pub fn compute(
        micro_transitions: &[(Vec<f64>, Vec<f64>)],
        macro_transitions: &[(Vec<f64>, Vec<f64>)],
    ) -> Self {
        let micro_ei = EffectiveInformation::compute(micro_transitions);
        let macro_ei = EffectiveInformation::compute(macro_transitions);

        let emergence = macro_ei.ei_bits - micro_ei.ei_bits;
        let causally_emergent = emergence > 0.0;

        // Confidence based on sample sizes and CI overlap
        let min_samples = micro_ei.sample_size.min(macro_ei.sample_size) as f64;
        let sample_confidence = (min_samples / 100.0).clamp(0.0, 1.0);

        // Check if CIs are well-separated
        let ci_separation = if causally_emergent {
            (macro_ei.confidence_interval.0 - micro_ei.confidence_interval.1).max(0.0)
        } else {
            0.0
        };
        let ci_confidence = (ci_separation / macro_ei.ei_bits.max(0.001)).clamp(0.0, 1.0);

        let confidence = 0.5 * sample_confidence + 0.5 * ci_confidence;

        Self {
            micro_ei,
            macro_ei,
            emergence,
            causally_emergent,
            confidence,
        }
    }

    /// Get interpretation of emergence level
    pub fn interpretation(&self) -> EmergenceInterpretation {
        if self.emergence > 0.5 && self.confidence > 0.7 {
            EmergenceInterpretation::StrongEmergence
        } else if self.emergence > 0.1 && self.confidence > 0.5 {
            EmergenceInterpretation::ModerateEmergence
        } else if self.emergence > 0.0 {
            EmergenceInterpretation::WeakEmergence
        } else if self.emergence > -0.1 {
            EmergenceInterpretation::Neutral
        } else {
            EmergenceInterpretation::Reductive
        }
    }
}

/// Interpretation of causal emergence level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceInterpretation {
    /// Strong emergence: Consciousness has significantly more causal power
    StrongEmergence,
    /// Moderate emergence: Consciousness has notable causal power
    ModerateEmergence,
    /// Weak emergence: Small causal advantage for consciousness
    WeakEmergence,
    /// Neutral: No significant difference
    Neutral,
    /// Reductive: Micro-level has more causal power (consciousness is epiphenomenal)
    Reductive,
}

// ═══════════════════════════════════════════════════════════════════════════
// CAUSAL VALIDATED ROUTER
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for causal emergence-validated routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalValidatedConfig {
    /// Minimum emergence for conscious routing
    pub min_emergence: f64,
    /// Minimum confidence for switching modes
    pub min_confidence: f64,
    /// Window size for CE estimation
    pub window_size: usize,
    /// Update interval (how often to recompute CE)
    pub update_interval: usize,
    /// Enable adaptive mode switching
    pub adaptive_mode: bool,
    /// Fallback strategy when CE <= 0
    pub fallback_strategy: RoutingStrategy,
}

impl Default for CausalValidatedConfig {
    fn default() -> Self {
        Self {
            min_emergence: 0.05,
            min_confidence: 0.5,
            window_size: 100,
            update_interval: 10,
            adaptive_mode: true,
            fallback_strategy: RoutingStrategy::HeuristicGuided,
        }
    }
}

/// Statistics for causal emergence routing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalValidatedStats {
    /// Total routing decisions
    pub decisions_made: u64,
    /// Decisions using conscious routing (CE > 0)
    pub conscious_decisions: u64,
    /// Decisions using fallback (CE <= 0)
    pub fallback_decisions: u64,
    /// Average emergence value
    pub avg_emergence: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Times emergence increased
    pub emergence_increases: u64,
    /// Times emergence decreased
    pub emergence_decreases: u64,
}

impl CausalValidatedStats {
    /// Get conscious routing ratio
    pub fn conscious_ratio(&self) -> f64 {
        if self.decisions_made == 0 {
            0.0
        } else {
            self.conscious_decisions as f64 / self.decisions_made as f64
        }
    }
}

/// Recorded transition for CE computation
#[derive(Debug, Clone)]
struct RecordedTransition {
    /// Micro-level state (primitives, neurons)
    micro_state: Vec<f64>,
    /// Macro-level state (consciousness features)
    macro_state: Vec<f64>,
    /// Next micro-level state
    next_micro_state: Vec<f64>,
    /// Next macro-level state
    next_macro_state: Vec<f64>,
}

/// Current routing mode based on causal emergence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRoutingMode {
    /// Using full conscious routing (CE > threshold)
    ConsciousRouting,
    /// Using fallback routing (CE <= threshold)
    FallbackRouting,
    /// Gathering data (not enough samples yet)
    Calibrating,
}

/// The Causal Emergence-Validated Router
///
/// Uses Causal Emergence to validate and adapt consciousness-guided routing.
/// When CE > threshold, uses full conscious routing via OscillatoryRouter.
/// Otherwise, falls back to simpler strategies.
pub struct CausalValidatedRouter {
    /// Inner oscillatory router (real implementation as of Phase 5G)
    oscillatory_router: OscillatoryRouter,
    /// Current causal emergence measurement
    current_ce: Option<CausalEmergence>,
    /// Configuration
    config: CausalValidatedConfig,
    /// Statistics
    stats: CausalValidatedStats,
    /// Transition history for CE computation
    transition_history: VecDeque<RecordedTransition>,
    /// Cycle counter for update timing
    cycle_counter: u64,
    /// Current routing mode
    current_mode: CausalRoutingMode,
}

impl CausalValidatedRouter {
    /// Create new causal emergence-validated router
    pub fn new(config: CausalValidatedConfig) -> Self {
        Self {
            oscillatory_router: OscillatoryRouter::new(OscillatoryRouterConfig::default()),
            current_ce: None,
            config,
            stats: CausalValidatedStats::default(),
            transition_history: VecDeque::with_capacity(200),
            cycle_counter: 0,
            current_mode: CausalRoutingMode::Calibrating,
        }
    }

    /// Record a state transition for CE computation
    pub fn record_transition(
        &mut self,
        micro_state: Vec<f64>,
        macro_state: Vec<f64>,
        next_micro_state: Vec<f64>,
        next_macro_state: Vec<f64>,
    ) {
        let transition = RecordedTransition {
            micro_state,
            macro_state,
            next_micro_state,
            next_macro_state,
        };

        self.transition_history.push_back(transition);
        while self.transition_history.len() > self.config.window_size {
            self.transition_history.pop_front();
        }
    }

    /// Compute current causal emergence from history
    fn compute_causal_emergence(&self) -> Option<CausalEmergence> {
        if self.transition_history.len() < 20 {
            return None;
        }

        let micro_transitions: Vec<_> = self.transition_history.iter()
            .map(|t| (t.micro_state.clone(), t.next_micro_state.clone()))
            .collect();

        let macro_transitions: Vec<_> = self.transition_history.iter()
            .map(|t| (t.macro_state.clone(), t.next_macro_state.clone()))
            .collect();

        Some(CausalEmergence::compute(&micro_transitions, &macro_transitions))
    }

    /// Update routing mode based on current CE
    fn update_mode(&mut self) {
        let old_mode = self.current_mode;

        if let Some(ref ce) = self.current_ce {
            if ce.emergence > self.config.min_emergence && ce.confidence > self.config.min_confidence {
                self.current_mode = CausalRoutingMode::ConsciousRouting;
            } else if ce.confidence > self.config.min_confidence {
                self.current_mode = CausalRoutingMode::FallbackRouting;
            } else {
                self.current_mode = CausalRoutingMode::Calibrating;
            }

            // Track emergence trends
            if old_mode != self.current_mode {
                if self.current_mode == CausalRoutingMode::ConsciousRouting {
                    self.stats.emergence_increases += 1;
                } else if old_mode == CausalRoutingMode::ConsciousRouting {
                    self.stats.emergence_decreases += 1;
                }
            }
        }
    }

    /// Route with causal emergence validation
    pub fn route_validated(
        &mut self,
        current_state: &LatentConsciousnessState,
    ) -> ValidatedRoutingDecision {
        self.stats.decisions_made += 1;

        match self.current_mode {
            CausalRoutingMode::ConsciousRouting => {
                self.stats.conscious_decisions += 1;
                let plan = self.oscillatory_router.plan_phase_locked(current_state);

                ValidatedRoutingDecision {
                    strategy: plan.combined_strategy.magnitude_strategy,
                    mode: self.current_mode,
                    emergence: self.current_ce.as_ref().map(|c| c.emergence).unwrap_or(0.0),
                    confidence: self.current_ce.as_ref().map(|c| c.confidence).unwrap_or(0.0),
                    oscillatory_plan: Some(plan),
                    causal_validation: self.current_ce.clone(),
                }
            }
            CausalRoutingMode::FallbackRouting => {
                self.stats.fallback_decisions += 1;

                ValidatedRoutingDecision {
                    strategy: self.config.fallback_strategy,
                    mode: self.current_mode,
                    emergence: self.current_ce.as_ref().map(|c| c.emergence).unwrap_or(0.0),
                    confidence: self.current_ce.as_ref().map(|c| c.confidence).unwrap_or(0.0),
                    oscillatory_plan: None,
                    causal_validation: self.current_ce.clone(),
                }
            }
            CausalRoutingMode::Calibrating => {
                // Use conservative strategy while calibrating
                ValidatedRoutingDecision {
                    strategy: RoutingStrategy::StandardProcessing,
                    mode: self.current_mode,
                    emergence: 0.0,
                    confidence: 0.0,
                    oscillatory_plan: None,
                    causal_validation: None,
                }
            }
        }
    }

    /// Run one cycle of the validated router
    pub fn cycle(&mut self, dt: f64) {
        self.cycle_counter += 1;

        // Advance inner router
        self.oscillatory_router.cycle(dt);

        // Periodically update CE
        if self.cycle_counter % self.config.update_interval as u64 == 0 {
            if let Some(ce) = self.compute_causal_emergence() {
                // Update running averages
                let n = self.stats.decisions_made as f64;
                if n > 0.0 {
                    self.stats.avg_emergence =
                        (self.stats.avg_emergence * (n - 1.0) + ce.emergence) / n;
                    self.stats.avg_confidence =
                        (self.stats.avg_confidence * (n - 1.0) + ce.confidence) / n;
                }

                self.current_ce = Some(ce);
                if self.config.adaptive_mode {
                    self.update_mode();
                }
            }
        }
    }

    /// Get current causal emergence
    pub fn causal_emergence(&self) -> Option<&CausalEmergence> {
        self.current_ce.as_ref()
    }

    /// Get current routing mode
    pub fn current_mode(&self) -> CausalRoutingMode {
        self.current_mode
    }

    /// Get statistics
    pub fn stats(&self) -> &CausalValidatedStats {
        &self.stats
    }

    /// Get summary
    pub fn summary(&self) -> CausalValidatedSummary {
        CausalValidatedSummary {
            mode: self.current_mode,
            emergence: self.current_ce.as_ref().map(|c| c.emergence).unwrap_or(0.0),
            interpretation: self.current_ce.as_ref()
                .map(|c| c.interpretation())
                .unwrap_or(EmergenceInterpretation::Neutral),
            conscious_ratio: self.stats.conscious_ratio(),
            avg_emergence: self.stats.avg_emergence,
            avg_confidence: self.stats.avg_confidence,
            cycles: self.cycle_counter,
            oscillatory_summary: self.oscillatory_router.summary(),
        }
    }
}

/// Validated routing decision with causal emergence info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedRoutingDecision {
    /// Final routing strategy
    pub strategy: RoutingStrategy,
    /// Current routing mode
    pub mode: CausalRoutingMode,
    /// Current emergence value
    pub emergence: f64,
    /// Current confidence
    pub confidence: f64,
    /// Oscillatory plan (if using conscious routing)
    pub oscillatory_plan: Option<PhaseLockedPlan>,
    /// Causal validation details
    pub causal_validation: Option<CausalEmergence>,
}

/// Summary of causal validated router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalValidatedSummary {
    pub mode: CausalRoutingMode,
    pub emergence: f64,
    pub interpretation: EmergenceInterpretation,
    pub conscious_ratio: f64,
    pub avg_emergence: f64,
    pub avg_confidence: f64,
    pub cycles: u64,
    pub oscillatory_summary: OscillatoryRouterSummary,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_information_empty() {
        let ei = EffectiveInformation::compute(&[]);
        assert_eq!(ei.ei_bits, 0.0);
        assert_eq!(ei.sample_size, 0);
    }

    #[test]
    fn test_effective_information_basic() {
        let transitions: Vec<(Vec<f64>, Vec<f64>)> = (0..50)
            .map(|i| {
                let v = i as f64 / 50.0;
                (vec![v, v], vec![v * 0.9, v * 1.1])
            })
            .collect();

        let ei = EffectiveInformation::compute(&transitions);

        assert!(ei.ei_bits >= 0.0);
        assert_eq!(ei.sample_size, 50);
    }

    #[test]
    fn test_causal_emergence_computation() {
        let micro_transitions: Vec<(Vec<f64>, Vec<f64>)> = (0..50)
            .map(|i| {
                let v = i as f64 / 50.0;
                let noise = (i % 7) as f64 / 10.0;
                (vec![v, v + noise], vec![v * 0.8 + noise, v * 1.2])
            })
            .collect();

        let macro_transitions: Vec<(Vec<f64>, Vec<f64>)> = (0..50)
            .map(|i| {
                let v = i as f64 / 50.0;
                (vec![v], vec![v * 0.95])
            })
            .collect();

        let ce = CausalEmergence::compute(&micro_transitions, &macro_transitions);

        assert!(ce.micro_ei.ei_bits >= 0.0);
        assert!(ce.macro_ei.ei_bits >= 0.0);
    }

    #[test]
    fn test_emergence_interpretation() {
        let ce = CausalEmergence {
            micro_ei: EffectiveInformation {
                ei_bits: 1.0,
                sample_size: 100,
                confidence_interval: (0.8, 1.2),
                determinism: 0.5,
                degeneracy: 0.3,
            },
            macro_ei: EffectiveInformation {
                ei_bits: 1.6,
                sample_size: 100,
                confidence_interval: (1.4, 1.8),
                determinism: 0.8,
                degeneracy: 0.1,
            },
            emergence: 0.6,
            causally_emergent: true,
            confidence: 0.8,
        };

        assert_eq!(ce.interpretation(), EmergenceInterpretation::StrongEmergence);
    }

    #[test]
    fn test_causal_validated_router_creation() {
        let router = CausalValidatedRouter::new(CausalValidatedConfig::default());

        assert_eq!(router.current_mode, CausalRoutingMode::Calibrating);
        assert_eq!(router.stats.decisions_made, 0);
    }

    #[test]
    fn test_causal_validated_stats() {
        let mut stats = CausalValidatedStats::default();

        stats.decisions_made = 100;
        stats.conscious_decisions = 75;
        stats.fallback_decisions = 25;

        assert_eq!(stats.conscious_ratio(), 0.75);
    }
}
