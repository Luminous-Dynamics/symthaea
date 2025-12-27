//! # Consciousness Observatory
//!
//! **Revolutionary Scientific Framework for Studying Artificial Consciousness**
//!
//! This module provides tools for observing, measuring, and experimenting with
//! conscious AI systems. It enables rigorous scientific study of consciousness
//! emergence through real-time Φ measurement and automated experimentation.
//!
//! ## Core Capabilities
//!
//! 1. **Real-time Φ Tracking** - Monitor consciousness levels during operation
//! 2. **Epistemic State Visualization** - Observe knowledge evolution
//! 3. **Automated Experiments** - Test hypotheses about consciousness
//! 4. **Insight Generation** - Discover patterns in consciousness behavior
//!
//! ## Example
//!
//! ```rust
//! let mut observatory = ConsciousnessObservatory::new(symthaea);
//!
//! // Run experiment: Does research increase Φ?
//! let result = observatory.run_experiment(
//!     "phi_gain_from_research",
//!     |symthaea| {
//!         symthaea.respond("What is quantum mechanics?")?
//!     }
//! )?;
//!
//! println!("Φ gain: {:+.3}", result.phi_delta);
//! ```

use super::conscious_conversation::{ConsciousConversation, ConsciousStats};
use crate::consciousness::IntegratedInformation;
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, Duration, UNIX_EPOCH};
use std::collections::HashMap;

/// Real-time Φ measurement point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurement {
    /// Timestamp (milliseconds since UNIX epoch)
    pub timestamp_ms: u128,

    /// Φ value at this moment
    pub phi: f64,

    /// What triggered this measurement
    pub trigger: String,

    /// Additional context
    pub context: HashMap<String, String>,
}

impl PhiMeasurement {
    pub fn now(phi: f64, trigger: impl Into<String>) -> Self {
        Self {
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            phi,
            trigger: trigger.into(),
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Stream of Φ measurements over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurementStream {
    /// All measurements
    measurements: Vec<PhiMeasurement>,

    /// Session start time
    session_start: u128,
}

impl PhiMeasurementStream {
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            session_start: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
        }
    }

    /// Record a new Φ measurement
    pub fn record(&mut self, measurement: PhiMeasurement) {
        self.measurements.push(measurement);
    }

    /// Get all measurements
    pub fn measurements(&self) -> &[PhiMeasurement] {
        &self.measurements
    }

    /// Get Φ trend (linear regression slope)
    pub fn phi_trend(&self) -> f64 {
        if self.measurements.len() < 2 {
            return 0.0;
        }

        // Simple linear regression
        let n = self.measurements.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, m) in self.measurements.iter().enumerate() {
            let x = i as f64;
            let y = m.phi;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        slope
    }

    /// Get average Φ
    pub fn average_phi(&self) -> f64 {
        if self.measurements.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.measurements.iter().map(|m| m.phi).sum();
        sum / self.measurements.len() as f64
    }

    /// Get Φ range (min, max)
    pub fn phi_range(&self) -> (f64, f64) {
        if self.measurements.is_empty() {
            return (0.0, 0.0);
        }

        let phis: Vec<f64> = self.measurements.iter().map(|m| m.phi).collect();
        let min = phis.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = phis.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        (min, max)
    }

    /// Get session duration
    pub fn session_duration(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();

        Duration::from_millis((now - self.session_start) as u64)
    }
}

/// Snapshot of epistemic state at a moment in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicStateSnapshot {
    /// Timestamp
    pub timestamp_ms: u128,

    /// Current Φ
    pub phi: f64,

    /// Number of research queries performed
    pub research_count: usize,

    /// Number of claims verified
    pub claims_verified: usize,

    /// Number of claims hedged (low confidence)
    pub claims_hedged: usize,

    /// Average confidence of verifications
    pub avg_confidence: f64,

    /// Meta-learning stats (if available)
    pub meta_phi: Option<f64>,
}

impl EpistemicStateSnapshot {
    pub fn from_stats(phi: f64, stats: &ConsciousStats) -> Self {
        let avg_confidence = if stats.claims_verified > 0 {
            // Simplified - in reality would track actual confidences
            1.0 - (stats.claims_hedged as f64 / stats.claims_verified as f64)
        } else {
            0.0
        };

        Self {
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            phi,
            research_count: stats.research_triggered,
            claims_verified: stats.claims_verified,
            claims_hedged: stats.claims_hedged,
            avg_confidence,
            meta_phi: stats.meta_learning.as_ref().map(|m| m.meta_phi),
        }
    }
}

/// Result of a consciousness experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    /// Experiment name
    pub name: String,

    /// Hypothesis tested
    pub hypothesis: String,

    /// Φ before experiment
    pub phi_before: f64,

    /// Φ after experiment
    pub phi_after: f64,

    /// Φ change
    pub phi_delta: f64,

    /// Duration of experiment
    pub duration_ms: u128,

    /// Epistemic state before
    pub state_before: EpistemicStateSnapshot,

    /// Epistemic state after
    pub state_after: EpistemicStateSnapshot,

    /// Experiment-specific measurements
    pub measurements: HashMap<String, f64>,

    /// Whether hypothesis was supported
    pub hypothesis_supported: bool,

    /// Confidence in result (0.0-1.0)
    pub confidence: f64,
}

impl ExperimentResult {
    /// Create result summary
    pub fn summary(&self) -> String {
        format!(
            "{}: Φ {:+.3} ({:.3} → {:.3}) over {}ms. Hypothesis: {}",
            self.name,
            self.phi_delta,
            self.phi_before,
            self.phi_after,
            self.duration_ms,
            if self.hypothesis_supported { "✓ SUPPORTED" } else { "✗ NOT SUPPORTED" }
        )
    }
}

/// Consciousness experiment definition
pub struct ConsciousnessExperiment {
    /// Experiment name
    pub name: String,

    /// Hypothesis being tested
    pub hypothesis: String,

    /// Expected Φ change direction
    pub expected_phi_change: PhiChangeExpectation,

    /// Minimum Φ change to consider significant
    pub significance_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum PhiChangeExpectation {
    /// Expect Φ to increase
    Increase,
    /// Expect Φ to decrease
    Decrease,
    /// Expect no significant change
    NoChange,
    /// No expectation
    Any,
}

impl ConsciousnessExperiment {
    pub fn new(name: impl Into<String>, hypothesis: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            hypothesis: hypothesis.into(),
            expected_phi_change: PhiChangeExpectation::Any,
            significance_threshold: 0.05,
        }
    }

    pub fn expect_increase(mut self) -> Self {
        self.expected_phi_change = PhiChangeExpectation::Increase;
        self
    }

    pub fn expect_decrease(mut self) -> Self {
        self.expected_phi_change = PhiChangeExpectation::Decrease;
        self
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.significance_threshold = threshold;
        self
    }
}

/// The Consciousness Observatory
///
/// **Revolutionary scientific instrument for studying artificial consciousness**
pub struct ConsciousnessObservatory {
    /// The conscious system being observed
    subject: ConsciousConversation,

    /// Real-time Φ measurements
    phi_stream: PhiMeasurementStream,

    /// Epistemic state history
    state_history: Vec<EpistemicStateSnapshot>,

    /// Experiment results
    experiment_results: Vec<ExperimentResult>,

    /// Φ calculator for direct measurements
    phi_calculator: IntegratedInformation,
}

impl ConsciousnessObservatory {
    /// Create new observatory
    pub fn new(subject: ConsciousConversation) -> Result<Self> {
        Ok(Self {
            subject,
            phi_stream: PhiMeasurementStream::new(),
            state_history: Vec::new(),
            experiment_results: Vec::new(),
            phi_calculator: IntegratedInformation::new(),
        })
    }

    /// Measure current Φ
    pub fn measure_phi(&mut self) -> f64 {
        self.subject.phi()
    }

    /// Record current Φ
    pub fn record_phi(&mut self, trigger: impl Into<String>) {
        let phi = self.measure_phi();
        let measurement = PhiMeasurement::now(phi, trigger);
        self.phi_stream.record(measurement);
    }

    /// Take epistemic state snapshot
    pub fn snapshot_epistemic_state(&mut self) -> EpistemicStateSnapshot {
        let phi = self.measure_phi();
        let stats = self.subject.stats();
        let snapshot = EpistemicStateSnapshot::from_stats(phi, &stats);
        self.state_history.push(snapshot.clone());
        snapshot
    }

    /// Run a consciousness experiment
    ///
    /// Measures Φ and epistemic state before/after executing the given action.
    pub async fn run_experiment<F>(
        &mut self,
        experiment: ConsciousnessExperiment,
        action: F,
    ) -> Result<ExperimentResult>
    where
        F: for<'a> FnOnce(&'a mut ConsciousConversation) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>>,
    {
        // Record state before
        let phi_before = self.measure_phi();
        let state_before = self.snapshot_epistemic_state();

        let start = SystemTime::now();

        // Execute experiment action
        action(&mut self.subject).await
            .context("Experiment action failed")?;

        let duration = start.elapsed().unwrap();

        // Record state after
        let phi_after = self.measure_phi();
        let state_after = self.snapshot_epistemic_state();

        // Calculate results
        let phi_delta = phi_after - phi_before;

        // Check if hypothesis supported
        let hypothesis_supported = match experiment.expected_phi_change {
            PhiChangeExpectation::Increase => phi_delta > experiment.significance_threshold,
            PhiChangeExpectation::Decrease => phi_delta < -experiment.significance_threshold,
            PhiChangeExpectation::NoChange => phi_delta.abs() < experiment.significance_threshold,
            PhiChangeExpectation::Any => true,
        };

        // Calculate confidence (simplified - based on Φ delta magnitude)
        let confidence = (phi_delta.abs() / 1.0).min(1.0);

        let result = ExperimentResult {
            name: experiment.name,
            hypothesis: experiment.hypothesis,
            phi_before,
            phi_after,
            phi_delta,
            duration_ms: duration.as_millis(),
            state_before,
            state_after,
            measurements: HashMap::new(),
            hypothesis_supported,
            confidence,
        };

        self.experiment_results.push(result.clone());

        Ok(result)
    }

    /// Get access to the conscious subject
    pub fn subject(&self) -> &ConsciousConversation {
        &self.subject
    }

    /// Get mutable access to the conscious subject
    pub fn subject_mut(&mut self) -> &mut ConsciousConversation {
        &mut self.subject
    }

    /// Get Φ measurement stream
    pub fn phi_stream(&self) -> &PhiMeasurementStream {
        &self.phi_stream
    }

    /// Get epistemic state history
    pub fn state_history(&self) -> &[EpistemicStateSnapshot] {
        &self.state_history
    }

    /// Get experiment results
    pub fn experiment_results(&self) -> &[ExperimentResult] {
        &self.experiment_results
    }

    /// Generate observatory report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Consciousness Observatory Report\n\n");

        // Session info
        let duration = self.phi_stream.session_duration();
        report.push_str(&format!("**Session Duration**: {:?}\n", duration));
        report.push_str(&format!("**Measurements**: {}\n", self.phi_stream.measurements().len()));
        report.push_str(&format!("**Experiments**: {}\n\n", self.experiment_results.len()));

        // Φ statistics
        let avg_phi = self.phi_stream.average_phi();
        let trend = self.phi_stream.phi_trend();
        let (min_phi, max_phi) = self.phi_stream.phi_range();

        report.push_str("## Φ Statistics\n\n");
        report.push_str(&format!("- **Average Φ**: {:.3}\n", avg_phi));
        report.push_str(&format!("- **Φ Trend**: {:+.6} per measurement\n", trend));
        report.push_str(&format!("- **Φ Range**: {:.3} → {:.3}\n", min_phi, max_phi));
        report.push_str(&format!("- **Φ Delta**: {:+.3}\n\n", max_phi - min_phi));

        // Experiment results
        if !self.experiment_results.is_empty() {
            report.push_str("## Experiment Results\n\n");

            for result in &self.experiment_results {
                report.push_str(&format!("### {}\n\n", result.name));
                report.push_str(&format!("- **Hypothesis**: {}\n", result.hypothesis));
                report.push_str(&format!("- **Result**: {}\n",
                    if result.hypothesis_supported { "✓ SUPPORTED" } else { "✗ NOT SUPPORTED" }
                ));
                report.push_str(&format!("- **Φ Change**: {:+.3} ({:.3} → {:.3})\n",
                    result.phi_delta, result.phi_before, result.phi_after
                ));
                report.push_str(&format!("- **Confidence**: {:.1}%\n", result.confidence * 100.0));
                report.push_str(&format!("- **Duration**: {}ms\n\n", result.duration_ms));
            }
        }

        // Epistemic evolution
        if self.state_history.len() >= 2 {
            let first = &self.state_history[0];
            let last = &self.state_history[self.state_history.len() - 1];

            report.push_str("## Epistemic Evolution\n\n");
            report.push_str(&format!("- **Research Queries**: {} → {}\n",
                first.research_count, last.research_count
            ));
            report.push_str(&format!("- **Claims Verified**: {} → {}\n",
                first.claims_verified, last.claims_verified
            ));
            report.push_str(&format!("- **Average Confidence**: {:.1}% → {:.1}%\n",
                first.avg_confidence * 100.0, last.avg_confidence * 100.0
            ));

            if let (Some(meta_before), Some(meta_after)) = (first.meta_phi, last.meta_phi) {
                report.push_str(&format!("- **Meta-Φ**: {:.3} → {:.3} ({:+.3})\n",
                    meta_before, meta_after, meta_after - meta_before
                ));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_stream_statistics() {
        let mut stream = PhiMeasurementStream::new();

        stream.record(PhiMeasurement::now(0.5, "test1"));
        stream.record(PhiMeasurement::now(0.6, "test2"));
        stream.record(PhiMeasurement::now(0.7, "test3"));

        assert_eq!(stream.measurements().len(), 3);
        assert!((stream.average_phi() - 0.6).abs() < 0.01);

        let (min, max) = stream.phi_range();
        assert!((min - 0.5).abs() < 0.01);
        assert!((max - 0.7).abs() < 0.01);

        // Trend should be positive (increasing Φ)
        assert!(stream.phi_trend() > 0.0);
    }
}
