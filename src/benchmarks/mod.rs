// ==================================================================================
// Benchmark Suite for Symthaea
// ==================================================================================
//
// Allow dead code in experimental benchmark modules
#![allow(dead_code, unused_variables)]
//
// **Purpose**: Rigorous evaluation of Symthaea's capabilities vs baseline systems
//
// **Philosophy**: Claims must be backed by measurements. Symthaea's theoretical
// advantages (causal reasoning, temporal binding, consciousness) must translate
// to measurable performance gains on real tasks.
//
// **Benchmark Categories**:
//
// 1. **Causal Reasoning**: Where LLMs fundamentally fail
//    - Correlation vs causation detection
//    - Intervention prediction (do-calculus)
//    - Counterfactual reasoning
//    - Causal graph discovery
//
// 2. **Temporal Reasoning**: LTC advantage
//    - Irregular time series prediction
//    - Long-horizon forecasting
//    - Temporal binding/segmentation
//
// 3. **Compositional Generalization**: HDC advantage
//    - Novel combination understanding
//    - Zero-shot compositional transfer
//    - Systematic generalization
//
// 4. **Robustness**: Byzantine defense advantage
//    - Adversarial input detection
//    - Distribution shift handling
//    - Graceful degradation
//
// 5. **Consciousness Metrics**: Self-assessment
//    - Phi (Φ) measurement stability
//    - Workspace coherence
//    - Meta-cognitive accuracy
//
// ==================================================================================

pub mod causal_reasoning;
pub mod symthaea_solver;
pub mod compositional_benchmarks;
pub mod temporal_benchmarks;
pub mod robustness_benchmarks;
pub mod cladder_adapter;
pub mod cladder_nlp_adapter;
pub mod tuebingen_adapter;

// Re-export key types
pub use causal_reasoning::{
    CausalBenchmark,
    CausalBenchmarkSuite,
    CausalCategory,
    CausalGraph,
    CausalQuery,
    CausalAnswer,
    BenchmarkResults,
    BenchmarkResult,
    Observation,
};

pub use symthaea_solver::{SymthaeaSolver, SolverStats, run_symthaea_benchmarks};

pub use compositional_benchmarks::{
    CompositionalBenchmarkSuite,
    CompositionalSolver,
    CompositionalResults,
};

pub use temporal_benchmarks::{
    TemporalBenchmarkSuite,
    TemporalSolver,
    TemporalResults,
};

pub use robustness_benchmarks::{
    RobustnessBenchmarkSuite,
    RobustnessSolver,
    RobustnessResults,
};

pub use cladder_adapter::{
    CLadderAdapter,
    CLadderQuestion,
    CLadderResults,
};

pub use cladder_nlp_adapter::{
    CLadderNLPAdapter,
    ExtractedCausalRelation,
    ExtractedProbability,
    ExtractedQuestion,
};

pub use tuebingen_adapter::{
    TuebingenAdapter,
    TuebingenResults,
    CausalDirection,
    CauseEffectPair,
    discover_by_anm,
    discover_by_hsic,
    discover_combined,
    discover_by_nonlinear_anm,
    discover_by_igci,
    discover_by_reci,
    discover_combined_nonlinear,
    // Learned methods - self-contained implementation
    discover_by_learned,
    discover_enhanced_learned,
    create_learned_discoverer,
    LearnedCausalDiscovery,
    CausalDiscoveryResult,
    CrossValidatedLearner,
    CrossValidationResults,
    // HDC causal discovery - Symthaea's unique approach
    HdcCausalDiscovery,
    discover_by_hdc,
    discover_hdc_ensemble,
    discover_ultimate_ensemble,
    // Advanced HDC with all four improvements
    AdvancedHdcCausalDiscovery,
    TrainableHdcWeights,
    LtcCausalDynamics,
    CgnnStyleNetwork,
    DomainAwarePriors,
    CausalDomain,
    discover_advanced_hdc,
    discover_sota_ensemble,
    train_and_evaluate_advanced_hdc,
    // Information-theoretic methods (Phi-inspired)
    discover_by_conditional_entropy,
    discover_by_information_theoretic,
    discover_information_theoretic,
    // Smart ensemble (based on diagnostic analysis)
    discover_smart_ensemble,
    discover_majority_voting,
    // Principled Symthaea primitives (Dec 2025)
    PrincipledHdcCausal,
    discover_by_principled_hdc,
    PhiCausalDiscovery,
    discover_by_phi,
    discover_unified_primitives,
    discover_by_unified_primitives,
};

/// Run the full benchmark suite with Symthaea's real causal solver
pub fn run_all_benchmarks() -> BenchmarkReport {
    let mut report = BenchmarkReport::new();

    // Use real Symthaea solver!
    report.causal_results = Some(run_symthaea_benchmarks());

    // Run compositional benchmarks
    let comp_suite = CompositionalBenchmarkSuite::standard();
    let mut comp_solver = CompositionalSolver::new();
    report.compositional_results = Some(comp_suite.run(|b| comp_solver.solve(b)));

    // Run temporal benchmarks
    let temporal_suite = TemporalBenchmarkSuite::standard();
    let mut temporal_solver = TemporalSolver::new();
    report.temporal_results = Some(temporal_suite.run(|b| temporal_solver.solve(b)));

    // Run robustness benchmarks
    let robust_suite = RobustnessBenchmarkSuite::standard();
    let mut robust_solver = RobustnessSolver::new();
    report.robustness_results = Some(robust_suite.run(|b| robust_solver.solve(b)));

    report
}

/// Full benchmark report
#[derive(Debug)]
pub struct BenchmarkReport {
    pub causal_results: Option<BenchmarkResults>,
    pub compositional_results: Option<CompositionalResults>,
    pub temporal_results: Option<TemporalResults>,
    pub robustness_results: Option<RobustnessResults>,
}

impl BenchmarkReport {
    pub fn new() -> Self {
        Self {
            causal_results: None,
            compositional_results: None,
            temporal_results: None,
            robustness_results: None,
        }
    }

    /// Generate summary
    pub fn summary(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           SYMTHAEA BENCHMARK REPORT                          ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        if let Some(causal) = &self.causal_results {
            report.push_str(&causal.summary());
        }

        if let Some(comp) = &self.compositional_results {
            report.push_str("\n");
            report.push_str(&comp.summary());
        }

        if let Some(temporal) = &self.temporal_results {
            report.push_str("\n");
            report.push_str(&temporal.summary());
        }

        if let Some(robust) = &self.robustness_results {
            report.push_str("\n");
            report.push_str(&robust.summary());
        }

        // Overall summary
        report.push_str("\n╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                    OVERALL SUMMARY                           ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        let causal_acc = self.causal_results.as_ref().map(|r| r.accuracy() as f64).unwrap_or(0.0);
        let comp_acc = self.compositional_results.as_ref().map(|r| r.accuracy() as f64).unwrap_or(0.0);
        let temporal_acc = self.temporal_results.as_ref().map(|r| r.accuracy() as f64).unwrap_or(0.0);
        let robust_acc = self.robustness_results.as_ref().map(|r| r.accuracy() as f64).unwrap_or(0.0);

        report.push_str(&format!("  Causal Reasoning:     {:.1}%\n", causal_acc * 100.0));
        report.push_str(&format!("  Compositional:        {:.1}%\n", comp_acc * 100.0));
        report.push_str(&format!("  Temporal Reasoning:   {:.1}%\n", temporal_acc * 100.0));
        report.push_str(&format!("  Robustness:           {:.1}%\n", robust_acc * 100.0));
        report.push_str("  ────────────────────────────\n");

        let overall = (causal_acc + comp_acc + temporal_acc + robust_acc) / 4.0;
        report.push_str(&format!("  Overall Average:      {:.1}%\n\n", overall * 100.0));

        if overall >= 0.9 {
            report.push_str("  Status: EXCELLENT - Symthaea demonstrates strong AI capabilities!\n");
        } else if overall >= 0.7 {
            report.push_str("  Status: GOOD - Solid performance with room for improvement\n");
        } else {
            report.push_str("  Status: NEEDS WORK - Focus on failing categories\n");
        }

        report
    }
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_all_benchmarks() {
        let report = run_all_benchmarks();
        println!("{}", report.summary());

        assert!(report.causal_results.is_some());
    }
}
