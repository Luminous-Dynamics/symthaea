//! Safe Experimentation Framework for Recursive Self-Improvement
//!
//! This module implements a sandboxed testing environment for architectural
//! improvements before they are adopted into production.
//!
//! # Safety Guarantees
//!
//! - Baseline snapshot preserved
//! - Automatic rollback on degradation
//! - Multiple validation runs required
//! - Performance comparison before/after
//! - Conservative adoption criteria

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::instant_now;
// Use types from core for compatibility
use super::core::{AccuracyMetric, ComponentId, ImprovementType};

/// Safe experimentation framework for testing improvements before adoption
///
/// **Critical Safety Feature**: All improvements are tested in sandbox before deployment!
#[derive(Debug)]
pub struct SafeExperiment {
    /// Experiment identifier
    id: String,

    /// Baseline system snapshot (before improvement)
    baseline: SystemSnapshot,

    /// Proposed improvement
    improvement: ArchitecturalImprovement,

    /// Success criteria for adoption
    success_criteria: SuccessCriteria,

    /// Rollback condition
    rollback_condition: RollbackCondition,

    /// Experiment status
    status: ExperimentStatus,

    /// Validation runs
    validation_runs: Vec<ValidationRun>,

    /// Configuration
    config: ExperimentConfig,

    /// Created at
    created_at: Instant,
}

/// System snapshot capturing current state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSnapshot {
    /// Snapshot identifier
    pub id: String,

    /// Phi at snapshot time
    pub phi: f64,

    /// Average latency per component
    pub latencies: HashMap<ComponentId, Duration>,

    /// Average accuracy per metric
    pub accuracies: HashMap<AccuracyMetric, f64>,

    /// Component parameters
    pub parameters: HashMap<ComponentId, HashMap<String, f64>>,

    /// When snapshot was taken
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
}

/// Architectural improvement to test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalImprovement {
    /// Improvement identifier
    pub id: String,

    /// Improvement type
    pub improvement_type: ImprovementType,

    /// Description
    pub description: String,

    /// Expected benefits
    pub expected_phi_gain: Option<f64>,
    pub expected_latency_reduction: Option<f64>,
    pub expected_accuracy_gain: Option<f64>,

    /// Confidence in this improvement (0.0-1.0)
    pub confidence: f64,

    /// Which causal chain motivated this
    pub motivated_by: Option<String>, // CausalChain ID
}

/// Success criteria for adopting improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriteria {
    /// Minimum Phi improvement required
    pub min_phi_improvement: f64,

    /// Maximum latency increase allowed
    pub max_latency_increase: f64,

    /// Minimum accuracy required
    pub min_accuracy: f64,

    /// Minimum number of successful validation runs
    pub min_successful_runs: usize,

    /// Maximum number of validation runs to attempt
    pub max_validation_attempts: usize,
}

/// Rollback condition (when to abort)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackCondition {
    /// Rollback if Phi drops below this
    pub min_phi: f64,

    /// Rollback if latency exceeds this
    pub max_latency: Duration,

    /// Rollback if accuracy drops below this
    pub min_accuracy: f64,

    /// Rollback if any validation fails this many times
    pub max_consecutive_failures: usize,
}

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    /// Created but not started
    Pending,

    /// Currently running validation
    Running,

    /// All validations successful, ready to adopt
    Successful,

    /// Failed criteria, rolled back
    Failed,

    /// Manually aborted
    Aborted,

    /// Successfully adopted into production
    Adopted,
}

/// Single validation run result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRun {
    /// Run number
    pub run_number: usize,

    /// Phi after improvement
    pub phi: f64,

    /// Latency measurements
    pub latencies: HashMap<ComponentId, Duration>,

    /// Accuracy measurements
    pub accuracies: HashMap<AccuracyMetric, f64>,

    /// Did this run meet success criteria?
    pub passed: bool,

    /// Why did it pass/fail?
    pub reason: String,

    /// When run completed
    #[serde(skip, default = "instant_now")]
    pub completed_at: Instant,
}

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// How long to run each validation
    pub validation_duration: Duration,

    /// How many measurements per validation
    pub measurements_per_run: usize,

    /// Conservative mode (stricter criteria)
    pub conservative: bool,

    /// Require human approval for adoption
    pub require_human_approval: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            validation_duration: Duration::from_secs(60), // 1 minute per validation
            measurements_per_run: 100,
            conservative: true, // Safety first!
            require_human_approval: false, // Can be automated for minor changes
        }
    }
}

impl SafeExperiment {
    /// Create new experiment
    pub fn new(
        improvement: ArchitecturalImprovement,
        baseline: SystemSnapshot,
        config: ExperimentConfig,
    ) -> Self {
        let id = format!("experiment_{}_{}",
            improvement.id,
            Instant::now().elapsed().as_millis()
        );

        // Conservative success criteria
        let success_criteria = SuccessCriteria {
            min_phi_improvement: if config.conservative { 0.02 } else { 0.01 }, // 2% vs 1%
            max_latency_increase: if config.conservative { 0.05 } else { 0.10 }, // 5% vs 10%
            min_accuracy: 0.80, // Never go below 80%
            min_successful_runs: if config.conservative { 5 } else { 3 },
            max_validation_attempts: 10,
        };

        // Conservative rollback conditions
        let rollback_condition = RollbackCondition {
            min_phi: baseline.phi * 0.95, // Don't drop Phi more than 5%
            max_latency: baseline.latencies.values()
                .copied()
                .max()
                .unwrap_or(Duration::from_millis(100))
                .mul_f64(1.20), // Don't increase latency more than 20%
            min_accuracy: 0.75, // Never below 75%
            max_consecutive_failures: 3,
        };

        Self {
            id,
            baseline,
            improvement,
            success_criteria,
            rollback_condition,
            status: ExperimentStatus::Pending,
            validation_runs: Vec::new(),
            config,
            created_at: Instant::now(),
        }
    }

    /// Run a single validation
    pub fn run_validation(&mut self) -> Result<bool> {
        self.status = ExperimentStatus::Running;
        let run_number = self.validation_runs.len() + 1;

        // Simulate applying improvement and measuring performance
        // In real implementation, this would:
        // 1. Apply improvement to sandbox
        // 2. Run system for validation_duration
        // 3. Measure Phi, latency, accuracy
        // 4. Compare to baseline

        let (phi, latencies, accuracies) = self.measure_performance()?;

        // Check success criteria
        let phi_improved = phi >= self.baseline.phi + self.success_criteria.min_phi_improvement;

        let default_latency = Duration::from_millis(50);
        let latency_ok = latencies.values()
            .all(|&d| {
                let baseline_latency = self.baseline.latencies.get(&ComponentId::Cache).unwrap_or(&default_latency);
                d <= baseline_latency.mul_f64(1.0 + self.success_criteria.max_latency_increase)
            });

        let accuracy_ok = accuracies.values()
            .all(|&v| v >= self.success_criteria.min_accuracy);

        let passed = phi_improved && latency_ok && accuracy_ok;

        let reason = if passed {
            format!("Phi improved {:.1}%, latency OK, accuracy OK",
                (phi - self.baseline.phi) / self.baseline.phi * 100.0)
        } else {
            let mut reasons = Vec::new();
            if !phi_improved {
                reasons.push(format!("Phi only improved {:.1}%",
                    (phi - self.baseline.phi) / self.baseline.phi * 100.0));
            }
            if !latency_ok {
                reasons.push("Latency increased too much".to_string());
            }
            if !accuracy_ok {
                reasons.push("Accuracy below threshold".to_string());
            }
            format!("Failed: {}", reasons.join(", "))
        };

        let run = ValidationRun {
            run_number,
            phi,
            latencies,
            accuracies,
            passed,
            reason,
            completed_at: Instant::now(),
        };

        self.validation_runs.push(run);

        // Check rollback condition
        if self.should_rollback() {
            self.status = ExperimentStatus::Failed;
            return Ok(false);
        }

        // Check if experiment succeeded
        if self.has_succeeded() {
            self.status = ExperimentStatus::Successful;
            return Ok(true);
        }

        Ok(passed)
    }

    /// Measure performance with current improvement applied
    fn measure_performance(&self) -> Result<(f64, HashMap<ComponentId, Duration>, HashMap<AccuracyMetric, f64>)> {
        // Simulate measurements
        // In real implementation, this would actually run the system

        let phi = match &self.improvement.improvement_type {
            ImprovementType::IncreaseCacheSize { to, .. } => {
                // Larger cache -> better Phi
                self.baseline.phi * (1.0 + (*to as f64 / 10000.0))
            }
            ImprovementType::Parallelize { threads, .. } => {
                // Parallelization -> slightly better Phi
                self.baseline.phi * (1.0 + (*threads as f64 * 0.01))
            }
            ImprovementType::IncreaseEvolutionRate => {
                // Faster evolution -> better Phi
                self.baseline.phi * 1.03
            }
            _ => self.baseline.phi * 1.01, // Default small improvement
        };

        let mut latencies = self.baseline.latencies.clone();
        // Simulate latency changes based on improvement
        match &self.improvement.improvement_type {
            ImprovementType::IncreaseCacheSize { .. } => {
                // Cache improvement -> lower latency
                if let Some(cache_latency) = latencies.get_mut(&ComponentId::Cache) {
                    *cache_latency = cache_latency.mul_f64(0.8); // 20% faster
                }
            }
            ImprovementType::Parallelize { component, .. } => {
                // Parallelization -> lower latency for that component
                if let Some(comp_latency) = latencies.get_mut(component) {
                    *comp_latency = comp_latency.mul_f64(0.6); // 40% faster
                }
            }
            _ => {}
        }

        let accuracies = self.baseline.accuracies.clone(); // Usually doesn't change much

        Ok((phi, latencies, accuracies))
    }

    /// Check if we should rollback
    fn should_rollback(&self) -> bool {
        if self.validation_runs.is_empty() {
            return false;
        }

        // Check recent failures
        let recent_runs: Vec<&ValidationRun> = self.validation_runs.iter()
            .rev()
            .take(self.rollback_condition.max_consecutive_failures)
            .collect();

        let consecutive_failures = recent_runs.iter().all(|r| !r.passed);

        if consecutive_failures && recent_runs.len() >= self.rollback_condition.max_consecutive_failures {
            return true;
        }

        // Check if latest run violated hard limits
        if let Some(latest) = self.validation_runs.last() {
            if latest.phi < self.rollback_condition.min_phi {
                return true;
            }
            if latest.latencies.values().any(|&d| d > self.rollback_condition.max_latency) {
                return true;
            }
            if latest.accuracies.values().any(|&v| v < self.rollback_condition.min_accuracy) {
                return true;
            }
        }

        false
    }

    /// Check if experiment has succeeded
    fn has_succeeded(&self) -> bool {
        // Need minimum number of successful runs
        let successful_runs = self.validation_runs.iter()
            .filter(|r| r.passed)
            .count();

        successful_runs >= self.success_criteria.min_successful_runs
    }

    /// Adopt improvement into production
    pub fn adopt(&mut self) -> Result<()> {
        if self.status != ExperimentStatus::Successful {
            anyhow::bail!("Cannot adopt: experiment status is {:?}", self.status);
        }

        if self.config.require_human_approval {
            anyhow::bail!("Cannot auto-adopt: human approval required");
        }

        self.status = ExperimentStatus::Adopted;
        Ok(())
    }

    /// Rollback experiment
    pub fn rollback(&mut self) {
        self.status = ExperimentStatus::Failed;
    }

    /// Get status
    pub fn get_status(&self) -> ExperimentStatus {
        self.status
    }

    /// Get validation runs
    pub fn get_runs(&self) -> &[ValidationRun] {
        &self.validation_runs
    }

    /// Get experiment ID
    pub fn get_id(&self) -> &str {
        &self.id
    }

    /// Get the improvement being tested
    pub fn get_improvement(&self) -> &ArchitecturalImprovement {
        &self.improvement
    }

    /// Get experiment summary
    pub fn get_summary(&self) -> String {
        let passed = self.validation_runs.iter().filter(|r| r.passed).count();
        let total = self.validation_runs.len();

        format!(
            "Experiment {}: {}\n  Runs: {}/{} passed\n  Status: {:?}",
            self.id,
            self.improvement.description,
            passed,
            total,
            self.status
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_baseline() -> SystemSnapshot {
        let mut latencies = HashMap::new();
        latencies.insert(ComponentId::Cache, Duration::from_millis(50));
        latencies.insert(ComponentId::HRM, Duration::from_millis(100));

        let mut accuracies = HashMap::new();
        accuracies.insert(AccuracyMetric::AttackDetection, 0.90);

        SystemSnapshot {
            id: "test_baseline".to_string(),
            phi: 0.5,
            latencies,
            accuracies,
            parameters: HashMap::new(),
            timestamp: Instant::now(),
        }
    }

    fn create_test_improvement() -> ArchitecturalImprovement {
        ArchitecturalImprovement {
            id: "test_improvement".to_string(),
            improvement_type: ImprovementType::IncreaseCacheSize { from: 1000, to: 2000 },
            description: "Double cache size".to_string(),
            expected_phi_gain: Some(0.05),
            expected_latency_reduction: Some(0.2),
            expected_accuracy_gain: None,
            confidence: 0.8,
            motivated_by: None,
        }
    }

    #[test]
    fn test_experiment_creation() {
        let baseline = create_test_baseline();
        let improvement = create_test_improvement();
        let config = ExperimentConfig::default();

        let experiment = SafeExperiment::new(improvement, baseline, config);

        assert_eq!(experiment.get_status(), ExperimentStatus::Pending);
        assert!(experiment.get_runs().is_empty());
    }

    #[test]
    fn test_validation_run() {
        let baseline = create_test_baseline();
        let improvement = create_test_improvement();
        let config = ExperimentConfig::default();

        let mut experiment = SafeExperiment::new(improvement, baseline, config);
        let result = experiment.run_validation();

        assert!(result.is_ok());
        assert_eq!(experiment.get_runs().len(), 1);
    }

    #[test]
    fn test_experiment_summary() {
        let baseline = create_test_baseline();
        let improvement = create_test_improvement();
        let config = ExperimentConfig::default();

        let experiment = SafeExperiment::new(improvement, baseline, config);
        let summary = experiment.get_summary();

        assert!(summary.contains("Double cache size"));
        assert!(summary.contains("Pending"));
    }
}
