// Adaptive Programs
//
// Programs that adapt and re-synthesize themselves as causal structures change
//
// Innovation: Programs that monitor their own performance and update
// when the causal relationships in the environment change

use super::synthesizer::{CausalProgramSynthesizer, SynthesizedProgram, SynthesisConfig};
use super::verifier::{CounterfactualVerifier, VerificationResult, VerificationConfig};
use super::causal_spec::CausalSpec;
use crate::observability::ProbabilisticCausalGraph;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Strategy for adapting program
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AdaptationStrategy {
    /// Re-synthesize when verification fails
    OnVerificationFailure,

    /// Re-synthesize periodically
    Periodic { interval: usize },

    /// Re-synthesize when causal structure changes significantly
    OnCausalChange { threshold: f64 },

    /// Combination of strategies
    Hybrid,
}

/// Monitors program performance and triggers adaptation
pub struct ProgramMonitor {
    /// Recent verification results
    recent_results: VecDeque<VerificationResult>,

    /// Maximum number of results to keep
    max_history: usize,

    /// Threshold for triggering adaptation
    failure_threshold: f64,

    /// Number of iterations since last adaptation
    iterations_since_adaptation: usize,
}

impl ProgramMonitor {
    /// Create new program monitor
    pub fn new(max_history: usize, failure_threshold: f64) -> Self {
        Self {
            recent_results: VecDeque::with_capacity(max_history),
            max_history,
            failure_threshold,
            iterations_since_adaptation: 0,
        }
    }

    /// Record a verification result
    pub fn record(&mut self, result: VerificationResult) {
        if self.recent_results.len() >= self.max_history {
            self.recent_results.pop_front();
        }
        self.recent_results.push_back(result);
        self.iterations_since_adaptation += 1;
    }

    /// Check if adaptation should be triggered
    pub fn should_adapt(&self, strategy: AdaptationStrategy) -> bool {
        match strategy {
            AdaptationStrategy::OnVerificationFailure => {
                // Adapt if recent results show failure
                let recent_failures = self
                    .recent_results
                    .iter()
                    .filter(|r| !r.success)
                    .count();
                let failure_rate = recent_failures as f64 / self.recent_results.len() as f64;
                failure_rate > self.failure_threshold
            }

            AdaptationStrategy::Periodic { interval } => {
                self.iterations_since_adaptation >= interval
            }

            AdaptationStrategy::OnCausalChange { threshold } => {
                // Check if confidence has dropped
                if let Some(latest) = self.recent_results.back() {
                    latest.confidence < threshold
                } else {
                    false
                }
            }

            AdaptationStrategy::Hybrid => {
                // Combine multiple strategies
                self.should_adapt(AdaptationStrategy::OnVerificationFailure)
                    || self.should_adapt(AdaptationStrategy::Periodic { interval: 100 })
                    || self.should_adapt(AdaptationStrategy::OnCausalChange { threshold: 0.8 })
            }
        }
    }

    /// Reset iteration counter after adaptation
    pub fn reset(&mut self) {
        self.iterations_since_adaptation = 0;
    }

    /// Get average confidence from recent results
    pub fn average_confidence(&self) -> f64 {
        if self.recent_results.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.recent_results.iter().map(|r| r.confidence).sum();
        sum / self.recent_results.len() as f64
    }
}

impl Default for ProgramMonitor {
    fn default() -> Self {
        Self::new(10, 0.3)
    }
}

/// Adaptive Program
///
/// A program that:
/// 1. Monitors its own performance
/// 2. Detects when causal structure changes
/// 3. Re-synthesizes itself to maintain correctness
pub struct AdaptiveProgram {
    /// Current synthesized program
    program: SynthesizedProgram,

    /// Causal specification to maintain
    specification: CausalSpec,

    /// Current causal graph (updated with new data)
    causal_graph: Option<ProbabilisticCausalGraph>,

    /// Synthesizer for re-synthesis
    synthesizer: CausalProgramSynthesizer,

    /// Verifier for checking correctness
    verifier: CounterfactualVerifier,

    /// Monitor for tracking performance
    monitor: ProgramMonitor,

    /// Adaptation strategy
    strategy: AdaptationStrategy,

    /// Number of times program has been re-synthesized
    adaptation_count: usize,
}

impl AdaptiveProgram {
    /// Create new adaptive program
    pub fn new(
        initial_program: SynthesizedProgram,
        specification: CausalSpec,
        strategy: AdaptationStrategy,
    ) -> Self {
        Self {
            program: initial_program,
            specification,
            causal_graph: None,
            synthesizer: CausalProgramSynthesizer::new(SynthesisConfig::default()),
            verifier: CounterfactualVerifier::new(VerificationConfig::default()),
            monitor: ProgramMonitor::default(),
            strategy,
            adaptation_count: 0,
        }
    }

    /// Update program with new observations
    ///
    /// This is the main entry point for adaptation
    pub fn update(&mut self, new_graph: Option<ProbabilisticCausalGraph>) -> bool {
        // Update causal graph if provided
        if let Some(graph) = new_graph {
            self.causal_graph = Some(graph);
        }

        // Verify current program
        let verification = self.verifier.verify(&self.program);
        self.monitor.record(verification.clone());

        // Check if we should adapt
        if self.monitor.should_adapt(self.strategy) {
            return self.adapt();
        }

        false // No adaptation needed
    }

    /// Re-synthesize program
    fn adapt(&mut self) -> bool {
        // Attempt re-synthesis
        match self.synthesizer.synthesize(&self.specification) {
            Ok(new_program) => {
                // Verify new program is better
                let new_verification = self.verifier.verify(&new_program);

                if new_verification.confidence > self.program.confidence {
                    // Accept new program
                    self.program = new_program;
                    self.adaptation_count += 1;
                    self.monitor.reset();
                    true
                } else {
                    // Keep old program
                    false
                }
            }
            Err(_) => {
                // Re-synthesis failed, keep old program
                false
            }
        }
    }

    /// Get current program
    pub fn program(&self) -> &SynthesizedProgram {
        &self.program
    }

    /// Get adaptation statistics
    pub fn stats(&self) -> AdaptationStats {
        AdaptationStats {
            adaptation_count: self.adaptation_count,
            current_confidence: self.program.confidence,
            average_confidence: self.monitor.average_confidence(),
            iterations_since_adaptation: self.monitor.iterations_since_adaptation,
        }
    }

    /// Set new specification (triggers immediate re-synthesis)
    pub fn set_specification(&mut self, new_spec: CausalSpec) -> bool {
        self.specification = new_spec;
        self.adapt()
    }

    /// Force immediate adaptation
    pub fn force_adapt(&mut self) -> bool {
        self.adapt()
    }
}

/// Statistics about adaptive program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationStats {
    /// Number of times program has adapted
    pub adaptation_count: usize,

    /// Current program confidence
    pub current_confidence: f64,

    /// Average confidence over recent history
    pub average_confidence: f64,

    /// Iterations since last adaptation
    pub iterations_since_adaptation: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::synthesizer::ProgramTemplate;
    use std::collections::HashMap;

    #[test]
    fn test_monitor_creation() {
        let monitor = ProgramMonitor::new(10, 0.3);
        assert_eq!(monitor.max_history, 10);
        assert_eq!(monitor.failure_threshold, 0.3);
    }

    #[test]
    fn test_monitor_records_results() {
        let mut monitor = ProgramMonitor::new(5, 0.3);

        let result = VerificationResult {
            success: true,
            confidence: 0.9,
            counterfactual_accuracy: 0.9,
            tests_run: 100,
            edge_cases: vec![],
            details: None,
        };

        monitor.record(result);
        assert_eq!(monitor.recent_results.len(), 1);
        assert_eq!(monitor.average_confidence(), 0.9);
    }

    #[test]
    fn test_adaptation_on_verification_failure() {
        let mut monitor = ProgramMonitor::new(5, 0.5);

        // Record mostly failures
        for i in 0..5 {
            let result = VerificationResult {
                success: i < 1, // Only 1 success
                confidence: if i < 1 { 0.9 } else { 0.3 },
                counterfactual_accuracy: 0.5,
                tests_run: 100,
                edge_cases: vec![],
                details: None,
            };
            monitor.record(result);
        }

        // Should trigger adaptation (80% failure rate)
        assert!(monitor.should_adapt(AdaptationStrategy::OnVerificationFailure));
    }

    #[test]
    fn test_periodic_adaptation() {
        let mut monitor = ProgramMonitor::new(10, 0.3);

        // Simulate 100 iterations
        for _ in 0..100 {
            monitor.iterations_since_adaptation += 1;
        }

        assert!(monitor.should_adapt(AdaptationStrategy::Periodic { interval: 50 }));
    }

    #[test]
    fn test_adaptive_program_creation() {
        let program = SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights: HashMap::new(),
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
                strength: 0.5,
            },
            achieved_strength: 0.5,
            confidence: 0.9,
            complexity: 1,
            explanation: None,
            variables: vec!["a".to_string(), "b".to_string()],
        };

        let spec = CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.5,
        };

        let adaptive = AdaptiveProgram::new(
            program,
            spec,
            AdaptationStrategy::Hybrid,
        );

        assert_eq!(adaptive.adaptation_count, 0);
        assert_eq!(adaptive.program().confidence, 0.9);
    }

    #[test]
    fn test_adaptation_stats() {
        let program = SynthesizedProgram {
            template: ProgramTemplate::Linear {
                weights: HashMap::new(),
                bias: 0.0,
            },
            specification: CausalSpec::MakeCause {
                cause: "a".to_string(),
                effect: "b".to_string(),
                strength: 0.5,
            },
            achieved_strength: 0.5,
            confidence: 0.9,
            complexity: 1,
            explanation: None,
            variables: vec!["a".to_string(), "b".to_string()],
        };

        let spec = CausalSpec::MakeCause {
            cause: "a".to_string(),
            effect: "b".to_string(),
            strength: 0.5,
        };

        let adaptive = AdaptiveProgram::new(
            program,
            spec,
            AdaptationStrategy::Hybrid,
        );

        let stats = adaptive.stats();
        assert_eq!(stats.adaptation_count, 0);
        assert_eq!(stats.current_confidence, 0.9);
    }
}
