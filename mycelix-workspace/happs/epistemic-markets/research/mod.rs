//! Research Validation Infrastructure for Epistemic Markets
//!
//! This module provides comprehensive infrastructure for validating the research
//! questions outlined in RESEARCH_AGENDA.md. It includes:
//!
//! ## Research Areas
//!
//! ### F1: Aggregation Effectiveness (`aggregation_study`)
//! - Comparative analysis framework vs. expert prediction
//! - Simulation of different aggregation conditions
//! - Metrics: accuracy, calibration, resolution
//!
//! ### F2: Calibration Training (`calibration_training`)
//! - Training method effectiveness tracking
//! - Longitudinal study infrastructure
//! - Cross-domain transfer experiments
//! - Brier score tracking over time
//!
//! ### F3: Wisdom Seed Transmission (`wisdom_seeds`)
//! - Seed influence tracking
//! - A/B testing framework for seed formats
//! - Citation network analysis
//! - Transmission effectiveness metrics
//!
//! ### F4: Mechanism Design (`mechanism_comparison`)
//! - LMSR vs order book comparison
//! - Load testing harness
//! - Liquidity analysis
//! - Slippage measurements
//!
//! ### F5: Anti-Manipulation (`manipulation_detection`)
//! - Gradient poisoning attack simulation
//! - Collusion detection algorithms
//! - Sybil resistance testing
//! - Anomaly detection for market manipulation
//!
//! ## Common Infrastructure
//!
//! ### Metrics (`metrics`)
//! - Brier Score and decomposition
//! - Expected Calibration Error (ECE)
//! - Log scoring rules
//! - Statistical significance testing
//! - Time series analysis
//!
//! ### Simulation (`simulation`)
//! - Agent simulation with configurable behaviors
//! - Market dynamics simulation
//! - A/B testing framework
//! - Data collection and export
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use epistemic_markets::research::{
//!     aggregation_study::{AggregationStudy, AggregationStudyConfig},
//!     calibration_training::{CalibrationTrainingExperiment, TrainingExperimentConfig},
//!     metrics::BrierScore,
//! };
//!
//! // Run aggregation effectiveness study
//! let config = AggregationStudyConfig::default();
//! let mut study = AggregationStudy::new(config);
//! let results = study.run();
//! println!("Best aggregation method: {:?}", results.summary.best_overall_method);
//!
//! // Run calibration training experiment
//! let config = TrainingExperimentConfig::default();
//! let mut experiment = CalibrationTrainingExperiment::new(config);
//! let results = experiment.run();
//! println!("Calibration is trainable: {}", results.summary.calibration_is_trainable);
//! ```

// Core metrics for all research studies
pub mod metrics;

// Simulation framework
pub mod simulation;

// F1: Aggregation Effectiveness
pub mod aggregation_study;

// F2: Calibration Training
pub mod calibration_training;

// F3: Wisdom Seed Transmission
pub mod wisdom_seeds;

// F4: Mechanism Design
pub mod mechanism_comparison;

// F5: Anti-Manipulation
pub mod manipulation_detection;

// Re-exports for convenience
pub use aggregation_study::{
    AggregationMethod, AggregationStudy, AggregationStudyConfig, AggregationStudyResults,
    QuestionType,
};

pub use calibration_training::{
    CalibrationTrainingExperiment, TrainingExperimentConfig, TrainingExperimentResults,
    TrainingMethod,
};

pub use wisdom_seeds::{
    WisdomSeedFormat, WisdomStudyConfig, WisdomStudyResults, WisdomTransmissionStudy,
};

pub use mechanism_comparison::{
    MechanismComparisonConfig, MechanismComparisonResults, MechanismComparisonStudy, MechanismType,
    LMSRMarket, OrderBookMarket, LoadTestConfig, run_load_test,
};

pub use manipulation_detection::{
    ManipulationAttack, ManipulationSimConfig, ManipulationSimResults, ManipulationSimulation,
    DetectionMethod, SybilTestConfig, run_sybil_test,
};

pub use metrics::{
    BrierScore, BrierDecomposition, ExpectedCalibrationError, LogScore, InformationGain,
    StatisticalComparison, AggregationMetrics, TimeSeriesMetrics, DomainMetrics,
    TransferAnalysis, ExperimentSummary,
};

pub use simulation::{
    SimulationConfig, SimulationEngine, SimulationResults, SimulatedAgent, SimulatedMarket,
    AgentType, ExperimentGroup, Treatment, ABTestAnalysis,
};

/// Version of the research infrastructure
pub const RESEARCH_VERSION: &str = "0.1.0";

/// Run all quick validation tests
pub fn validate_infrastructure() -> ValidationReport {
    let mut report = ValidationReport::new();

    // Test metrics
    let predictions = vec![(0.7, 1.0), (0.3, 0.0), (0.6, 1.0), (0.4, 0.0)];
    let brier = BrierScore::calculate(&predictions);
    report.add_check("BrierScore calculation", brier.score.is_finite());

    // Test simulation
    let config = simulation::SimulationConfig {
        num_agents: 5,
        num_markets: 2,
        num_steps: 10,
        ..Default::default()
    };
    let mut engine = simulation::SimulationEngine::new(config);
    let results = engine.run();
    report.add_check("Simulation engine", results.total_steps > 0);

    // Test aggregation
    let preds = vec![(0.4, 1.0), (0.5, 1.0), (0.6, 1.0)];
    let agg = aggregation_study::aggregate_predictions(&preds, AggregationMethod::SimpleMean);
    report.add_check("Aggregation", (agg - 0.5).abs() < 0.001);

    // Test LMSR
    let market = mechanism_comparison::LMSRMarket::new("test".to_string(), 100.0, 2, 1000.0);
    let price = market.price(0);
    report.add_check("LMSR pricing", (price - 0.5).abs() < 0.01);

    report
}

/// Validation report for infrastructure checks
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub checks: Vec<(String, bool)>,
    pub all_passed: bool,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            checks: vec![],
            all_passed: true,
        }
    }

    pub fn add_check(&mut self, name: &str, passed: bool) {
        self.checks.push((name.to_string(), passed));
        if !passed {
            self.all_passed = false;
        }
    }

    pub fn summary(&self) -> String {
        let passed = self.checks.iter().filter(|(_, p)| *p).count();
        let total = self.checks.len();
        format!(
            "Validation: {}/{} checks passed ({})",
            passed,
            total,
            if self.all_passed { "OK" } else { "FAILED" }
        )
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infrastructure_validation() {
        let report = validate_infrastructure();
        assert!(report.all_passed, "Infrastructure validation failed: {:?}", report.checks);
    }

    #[test]
    fn test_metrics_import() {
        let brier = BrierScore::calculate(&[(0.5, 1.0), (0.5, 0.0)]);
        assert!(brier.score.is_finite());
    }

    #[test]
    fn test_aggregation_import() {
        let result = aggregation_study::aggregate_predictions(
            &[(0.3, 1.0), (0.7, 1.0)],
            AggregationMethod::SimpleMean,
        );
        assert!((result - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mechanism_import() {
        let market = LMSRMarket::new("test".to_string(), 100.0, 2, 1000.0);
        assert_eq!(market.quantities.len(), 2);
    }

    #[test]
    fn test_manipulation_import() {
        let attack = ManipulationAttack::WashTrading;
        assert_eq!(attack, ManipulationAttack::WashTrading);
    }
}
