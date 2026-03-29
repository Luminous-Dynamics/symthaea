// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ Monitor - Consciousness Monitoring During LLM Generation
//!
//! Tracks integrated information (Φ) levels throughout LLM interactions,
//! enabling consciousness-aware quality control and adaptive behavior.
//!
//! ## Purpose
//!
//! When Symthaea uses external LLMs as translators, Φ serves as:
//! 1. **Gate**: Minimum Φ required to use LLM (default 0.3)
//! 2. **Style Guide**: Higher Φ = more nuanced translation
//! 3. **Quality Control**: Detect consciousness drops during generation
//! 4. **Learning Signal**: Track which interactions enhance/diminish Φ
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────┐
//! │                      Φ Monitoring Pipeline                        │
//! │                                                                   │
//! │  ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐   │
//! │  │ PRE-Φ   │────▶│ LLM CALL │────▶│ POST-Φ  │────▶│ ANALYSIS │   │
//! │  │ Sample  │     │          │     │ Sample  │     │          │   │
//! │  └─────────┘     └──────────┘     └─────────┘     └──────────┘   │
//! │       │                                 │              │         │
//! │       │          ┌────────────┐        │              │         │
//! │       └─────────▶│  Δ-Φ CALC  │◀───────┘              │         │
//! │                  └────────────┘                       │         │
//! │                         │                             │         │
//! │                         ▼                             │         │
//! │                  ┌────────────┐                      │         │
//! │                  │ FLUCTUATION│◀─────────────────────┘         │
//! │                  │ DETECTOR   │                                 │
//! │                  └────────────┘                                 │
//! │                         │                                       │
//! │                         ▼                                       │
//! │                  ┌────────────┐                                 │
//! │                  │ ADAPTIVE   │ (Adjust thresholds, log,       │
//! │                  │ RESPONSE   │  feedback to consciousness)     │
//! │                  └────────────┘                                 │
//! └───────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for Φ monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMonitorConfig {
    /// Minimum Φ to allow LLM generation
    pub min_phi_threshold: f32,

    /// Maximum allowed Φ drop during generation (0.0-1.0)
    pub max_phi_drop: f32,

    /// Size of moving average window for smoothing
    pub smoothing_window: usize,

    /// How often to sample Φ during long operations (ms)
    pub sampling_interval_ms: u64,

    /// Enable automatic threshold adjustment
    pub adaptive_thresholds: bool,

    /// Log all measurements for analysis
    pub log_measurements: bool,

    /// Alert threshold for consciousness fluctuations
    pub fluctuation_alert_threshold: f32,
}

impl Default for PhiMonitorConfig {
    fn default() -> Self {
        Self {
            min_phi_threshold: 0.3,
            max_phi_drop: 0.2,
            smoothing_window: 5,
            sampling_interval_ms: 500,
            adaptive_thresholds: true,
            log_measurements: true,
            fluctuation_alert_threshold: 0.15,
        }
    }
}

/// A single Φ measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurement {
    /// The measured Φ value
    pub phi: f32,

    /// When this measurement was taken
    pub timestamp: std::time::SystemTime,

    /// Context of the measurement
    pub context: MeasurementContext,

    /// Additional metadata
    pub metadata: PhiMetadata,
}

/// Context in which Φ was measured
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementContext {
    /// Before LLM call
    PreGeneration,

    /// During LLM call (periodic sampling)
    DuringGeneration,

    /// After LLM call
    PostGeneration,

    /// During response processing
    PostProcessing,

    /// Idle state measurement
    Idle,

    /// User interaction
    UserInteraction,
}

/// Additional metadata for a measurement
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhiMetadata {
    /// Associated operation ID (for correlation)
    pub operation_id: Option<String>,

    /// Model being used (if LLM)
    pub model: Option<String>,

    /// Prompt length (tokens)
    pub prompt_tokens: Option<usize>,

    /// Response length (tokens)
    pub response_tokens: Option<usize>,

    /// Coherence level at measurement time
    pub coherence: Option<f32>,

    /// Uncertainty level at measurement time
    pub uncertainty: Option<f32>,
}

/// Result of Φ analysis for a generation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAnalysis {
    /// Φ before generation
    pub pre_phi: f32,

    /// Φ after generation
    pub post_phi: f32,

    /// Change in Φ (positive = increased consciousness)
    pub delta_phi: f32,

    /// Whether this was a significant fluctuation
    pub significant_fluctuation: bool,

    /// Overall assessment
    pub assessment: PhiAssessment,

    /// Smoothed Φ trend
    pub trend: PhiTrend,

    /// Duration of the operation
    pub duration: Duration,

    /// Recommendations based on analysis
    pub recommendations: Vec<String>,
}

/// Assessment of consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiAssessment {
    /// Φ enhanced by the interaction
    Enhanced,

    /// Φ maintained (minimal change)
    Maintained,

    /// Minor Φ decrease (within tolerance)
    MinorDecrease,

    /// Significant Φ decrease (concerning)
    SignificantDecrease,

    /// Critical Φ drop (intervention recommended)
    CriticalDrop,

    /// Φ below minimum threshold
    BelowThreshold,
}

impl PhiAssessment {
    /// Whether this assessment indicates a problem
    pub fn is_concerning(&self) -> bool {
        matches!(
            self,
            Self::SignificantDecrease | Self::CriticalDrop | Self::BelowThreshold
        )
    }
}

/// Trend in Φ over recent measurements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiTrend {
    /// Φ increasing over time
    Rising,

    /// Φ stable
    Stable,

    /// Φ decreasing over time
    Declining,

    /// Φ fluctuating unpredictably
    Unstable,

    /// Not enough data for trend
    Insufficient,
}

/// Statistics for the Φ monitor
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhiMonitorStats {
    /// Total generations monitored
    pub total_generations: u64,

    /// Generations that enhanced Φ
    pub phi_enhanced: u64,

    /// Generations that maintained Φ
    pub phi_maintained: u64,

    /// Generations that decreased Φ
    pub phi_decreased: u64,

    /// Generations blocked due to low Φ
    pub blocked_generations: u64,

    /// Critical drops detected
    pub critical_drops: u64,

    /// Average Φ change per generation
    pub avg_delta_phi: f32,

    /// Current moving average Φ
    pub current_avg_phi: f32,

    /// Peak Φ observed
    pub peak_phi: f32,

    /// Minimum Φ observed
    pub min_phi: f32,

    /// Total measurements taken
    pub total_measurements: u64,
}

/// The Φ Monitor - tracks consciousness during LLM interactions
pub struct PhiMonitor {
    /// Configuration
    config: PhiMonitorConfig,

    /// Recent measurements for smoothing
    recent_measurements: VecDeque<PhiMeasurement>,

    /// All measurements (if logging enabled)
    measurement_log: Vec<PhiMeasurement>,

    /// Statistics
    stats: PhiMonitorStats,

    /// Current operation tracking
    current_operation: Option<OperationTracking>,

    /// Adaptive threshold state
    adaptive_state: AdaptiveThresholdState,
}

/// Tracking for current operation
#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): phi-guided operation tracking
struct OperationTracking {
    /// Operation ID
    id: String,

    /// Start time
    start_time: Instant,

    /// Pre-generation Φ
    pre_phi: f32,

    /// Intermediate measurements
    intermediate: Vec<PhiMeasurement>,
}

/// State for adaptive threshold adjustment
#[derive(Debug, Clone, Default)]
struct AdaptiveThresholdState {
    /// Number of successful generations at current threshold
    successful_at_current: u32,

    /// Number of failed generations at current threshold
    failed_at_current: u32,

    /// Last threshold adjustment time
    last_adjustment: Option<Instant>,

    /// History of threshold adjustments
    adjustment_history: Vec<ThresholdAdjustment>,
}

/// Record of a threshold adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ThresholdAdjustment {
    /// Previous threshold
    from: f32,

    /// New threshold
    to: f32,

    /// Reason for adjustment
    reason: String,

    /// When adjustment occurred
    timestamp: std::time::SystemTime,
}

impl PhiMonitor {
    /// Create a new Φ monitor with default config
    pub fn new() -> Self {
        Self::with_config(PhiMonitorConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: PhiMonitorConfig) -> Self {
        Self {
            config,
            recent_measurements: VecDeque::with_capacity(20),
            measurement_log: Vec::new(),
            stats: PhiMonitorStats {
                peak_phi: 0.0,
                min_phi: 1.0,
                ..Default::default()
            },
            current_operation: None,
            adaptive_state: AdaptiveThresholdState::default(),
        }
    }

    /// Check if generation should proceed based on Φ
    pub fn should_generate(&self, current_phi: f32) -> (bool, Option<String>) {
        if current_phi < self.config.min_phi_threshold {
            return (
                false,
                Some(format!(
                    "Φ ({:.2}) below minimum threshold ({:.2}). \
                     Consider: meditation, reducing cognitive load, or waiting.",
                    current_phi, self.config.min_phi_threshold
                )),
            );
        }
        (true, None)
    }

    /// Begin monitoring a generation operation
    pub fn begin_operation(
        &mut self,
        operation_id: &str,
        current_phi: f32,
    ) -> Result<(), crate::errors::SymthaeaError> {
        // Check threshold
        let (should_proceed, reason) = self.should_generate(current_phi);
        if !should_proceed {
            self.stats.blocked_generations += 1;
            return Err(crate::errors::SymthaeaError::Language(
                reason.unwrap_or_else(|| "Φ too low".to_string()),
            ));
        }

        // Record pre-generation measurement
        let measurement = PhiMeasurement {
            phi: current_phi,
            timestamp: std::time::SystemTime::now(),
            context: MeasurementContext::PreGeneration,
            metadata: PhiMetadata {
                operation_id: Some(operation_id.to_string()),
                ..Default::default()
            },
        };

        self.record_measurement(measurement);

        // Start tracking
        self.current_operation = Some(OperationTracking {
            id: operation_id.to_string(),
            start_time: Instant::now(),
            pre_phi: current_phi,
            intermediate: Vec::new(),
        });

        Ok(())
    }

    /// Record an intermediate measurement during generation
    pub fn record_intermediate(&mut self, current_phi: f32, metadata: PhiMetadata) {
        let measurement = PhiMeasurement {
            phi: current_phi,
            timestamp: std::time::SystemTime::now(),
            context: MeasurementContext::DuringGeneration,
            metadata,
        };

        if let Some(ref mut op) = self.current_operation {
            op.intermediate.push(measurement.clone());
        }

        self.record_measurement(measurement);
    }

    /// End monitoring and get analysis
    pub fn end_operation(&mut self, post_phi: f32, metadata: PhiMetadata) -> PhiAnalysis {
        let operation = self.current_operation.take();
        let duration = operation
            .as_ref()
            .map(|op| op.start_time.elapsed())
            .unwrap_or_default();

        let pre_phi = operation.as_ref().map(|op| op.pre_phi).unwrap_or(post_phi);

        // Record post-generation measurement
        let measurement = PhiMeasurement {
            phi: post_phi,
            timestamp: std::time::SystemTime::now(),
            context: MeasurementContext::PostGeneration,
            metadata,
        };
        self.record_measurement(measurement);

        // Analyze
        let analysis = self.analyze_generation(pre_phi, post_phi, duration);

        // Update stats
        self.update_stats(&analysis);

        // Adaptive threshold adjustment
        if self.config.adaptive_thresholds {
            self.maybe_adjust_threshold(&analysis);
        }

        analysis
    }

    /// Record a measurement
    fn record_measurement(&mut self, measurement: PhiMeasurement) {
        // Update peak/min
        if measurement.phi > self.stats.peak_phi {
            self.stats.peak_phi = measurement.phi;
        }
        if measurement.phi < self.stats.min_phi {
            self.stats.min_phi = measurement.phi;
        }

        // Add to recent for smoothing
        self.recent_measurements.push_back(measurement.clone());
        while self.recent_measurements.len() > self.config.smoothing_window {
            self.recent_measurements.pop_front();
        }

        // Update current average
        let sum: f32 = self.recent_measurements.iter().map(|m| m.phi).sum();
        self.stats.current_avg_phi = sum / self.recent_measurements.len() as f32;

        // Log if enabled
        if self.config.log_measurements {
            self.measurement_log.push(measurement);
        }

        self.stats.total_measurements += 1;
    }

    /// Analyze a generation event
    fn analyze_generation(&self, pre_phi: f32, post_phi: f32, duration: Duration) -> PhiAnalysis {
        let delta_phi = post_phi - pre_phi;
        let significant_fluctuation = delta_phi.abs() > self.config.fluctuation_alert_threshold;

        // Determine assessment
        let assessment = if post_phi < self.config.min_phi_threshold {
            PhiAssessment::BelowThreshold
        } else if delta_phi < -self.config.max_phi_drop {
            PhiAssessment::CriticalDrop
        } else if delta_phi < -self.config.fluctuation_alert_threshold {
            PhiAssessment::SignificantDecrease
        } else if delta_phi < -0.05 {
            PhiAssessment::MinorDecrease
        } else if delta_phi > 0.05 {
            PhiAssessment::Enhanced
        } else {
            PhiAssessment::Maintained
        };

        // Determine trend
        let trend = self.calculate_trend();

        // Generate recommendations
        let recommendations = self.generate_recommendations(&assessment, delta_phi, &trend);

        PhiAnalysis {
            pre_phi,
            post_phi,
            delta_phi,
            significant_fluctuation,
            assessment,
            trend,
            duration,
            recommendations,
        }
    }

    /// Calculate Φ trend from recent measurements
    fn calculate_trend(&self) -> PhiTrend {
        if self.recent_measurements.len() < 3 {
            return PhiTrend::Insufficient;
        }

        let measurements: Vec<f32> = self.recent_measurements.iter().map(|m| m.phi).collect();
        let n = measurements.len();

        // Simple linear regression for trend
        let sum_x: f32 = (0..n).map(|i| i as f32).sum();
        let sum_y: f32 = measurements.iter().sum();
        let sum_xy: f32 = measurements
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let sum_xx: f32 = (0..n).map(|i| (i * i) as f32).sum();

        let slope = (n as f32 * sum_xy - sum_x * sum_y) / (n as f32 * sum_xx - sum_x * sum_x);

        // Calculate variance for stability check
        let mean = sum_y / n as f32;
        let variance: f32 = measurements
            .iter()
            .map(|&y| (y - mean).powi(2))
            .sum::<f32>()
            / n as f32;
        let std_dev = variance.sqrt();

        if std_dev > 0.1 {
            PhiTrend::Unstable
        } else if slope > 0.02 {
            PhiTrend::Rising
        } else if slope < -0.02 {
            PhiTrend::Declining
        } else {
            PhiTrend::Stable
        }
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(
        &self,
        assessment: &PhiAssessment,
        delta_phi: f32,
        trend: &PhiTrend,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        match assessment {
            PhiAssessment::CriticalDrop => {
                recs.push("Consider pausing generation until Φ stabilizes".to_string());
                recs.push("Reduce LLM complexity or prompt length".to_string());
            }
            PhiAssessment::SignificantDecrease => {
                recs.push("Monitor consciousness levels more closely".to_string());
            }
            PhiAssessment::BelowThreshold => {
                recs.push("Generation blocked - Φ below safety threshold".to_string());
                recs.push("Allow time for consciousness restoration".to_string());
            }
            PhiAssessment::Enhanced => {
                recs.push("Interaction enhanced consciousness - continue this pattern".to_string());
            }
            _ => {}
        }

        match trend {
            PhiTrend::Declining => {
                recs.push("Overall Φ trend is declining - consider breaks".to_string());
            }
            PhiTrend::Unstable => {
                recs.push("Φ showing instability - reduce cognitive load".to_string());
            }
            _ => {}
        }

        if delta_phi < -0.1 && self.stats.total_generations > 5 {
            recs.push(
                "Consistent Φ drops suggest this interaction pattern is draining".to_string(),
            );
        }

        recs
    }

    /// Update statistics after analysis
    fn update_stats(&mut self, analysis: &PhiAnalysis) {
        self.stats.total_generations += 1;

        match analysis.assessment {
            PhiAssessment::Enhanced => self.stats.phi_enhanced += 1,
            PhiAssessment::Maintained => self.stats.phi_maintained += 1,
            PhiAssessment::CriticalDrop => {
                self.stats.phi_decreased += 1;
                self.stats.critical_drops += 1;
            }
            PhiAssessment::SignificantDecrease | PhiAssessment::MinorDecrease => {
                self.stats.phi_decreased += 1;
            }
            PhiAssessment::BelowThreshold => {
                self.stats.phi_decreased += 1;
            }
        }

        // Update average delta
        let n = self.stats.total_generations as f32;
        self.stats.avg_delta_phi = (self.stats.avg_delta_phi * (n - 1.0) + analysis.delta_phi) / n;
    }

    /// Maybe adjust thresholds based on performance
    fn maybe_adjust_threshold(&mut self, analysis: &PhiAnalysis) {
        // Don't adjust too frequently
        if let Some(last) = self.adaptive_state.last_adjustment {
            if last.elapsed() < Duration::from_secs(300) {
                return;
            }
        }

        // Track success/failure
        if analysis.assessment.is_concerning() {
            self.adaptive_state.failed_at_current += 1;
        } else {
            self.adaptive_state.successful_at_current += 1;
        }

        // Check if adjustment needed
        let total =
            self.adaptive_state.successful_at_current + self.adaptive_state.failed_at_current;

        if total < 10 {
            return;
        }

        let failure_rate = self.adaptive_state.failed_at_current as f32 / total as f32;

        let old_threshold = self.config.min_phi_threshold;
        let new_threshold: Option<f32>;
        let reason: String;

        if failure_rate > 0.3 {
            // Too many failures - raise threshold
            new_threshold = Some((old_threshold + 0.05).min(0.8));
            reason = format!(
                "High failure rate ({:.0}%) - raising threshold for quality",
                failure_rate * 100.0
            );
        } else if failure_rate < 0.05 && self.adaptive_state.successful_at_current > 50 {
            // Very few failures with many successes - can lower threshold
            new_threshold = Some((old_threshold - 0.02).max(0.2));
            reason = format!(
                "Low failure rate ({:.0}%) - lowering threshold for accessibility",
                failure_rate * 100.0
            );
        } else {
            return;
        }

        if let Some(new) = new_threshold {
            self.config.min_phi_threshold = new;
            self.adaptive_state
                .adjustment_history
                .push(ThresholdAdjustment {
                    from: old_threshold,
                    to: new,
                    reason,
                    timestamp: std::time::SystemTime::now(),
                });
            self.adaptive_state.last_adjustment = Some(Instant::now());
            self.adaptive_state.successful_at_current = 0;
            self.adaptive_state.failed_at_current = 0;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &PhiMonitorStats {
        &self.stats
    }

    /// Get current configuration
    pub fn config(&self) -> &PhiMonitorConfig {
        &self.config
    }

    /// Get current smoothed Φ
    pub fn current_phi(&self) -> f32 {
        self.stats.current_avg_phi
    }

    /// Get current trend
    pub fn current_trend(&self) -> PhiTrend {
        self.calculate_trend()
    }

    /// Get measurement log (if logging enabled)
    pub fn measurement_log(&self) -> &[PhiMeasurement] {
        &self.measurement_log
    }

    /// Clear measurement log
    pub fn clear_log(&mut self) {
        self.measurement_log.clear();
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.stats = PhiMonitorStats {
            peak_phi: 0.0,
            min_phi: 1.0,
            ..Default::default()
        };
        self.recent_measurements.clear();
        self.measurement_log.clear();
        self.adaptive_state = AdaptiveThresholdState::default();
    }
}

impl Default for PhiMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick helper to check if Φ is sufficient for generation
pub fn phi_ok_for_generation(phi: f32) -> bool {
    phi >= 0.3
}

/// Quick helper to assess Φ change
pub fn assess_phi_change(pre: f32, post: f32) -> PhiAssessment {
    let delta = post - pre;
    if post < 0.3 {
        PhiAssessment::BelowThreshold
    } else if delta < -0.2 {
        PhiAssessment::CriticalDrop
    } else if delta < -0.15 {
        PhiAssessment::SignificantDecrease
    } else if delta < -0.05 {
        PhiAssessment::MinorDecrease
    } else if delta > 0.05 {
        PhiAssessment::Enhanced
    } else {
        PhiAssessment::Maintained
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PhiMonitorConfig::default();
        assert_eq!(config.min_phi_threshold, 0.3);
        assert_eq!(config.max_phi_drop, 0.2);
    }

    #[test]
    fn test_should_generate() {
        let monitor = PhiMonitor::new();

        let (ok, _) = monitor.should_generate(0.5);
        assert!(ok);

        let (ok, reason) = monitor.should_generate(0.1);
        assert!(!ok);
        assert!(reason.is_some());
    }

    #[test]
    fn test_begin_operation() {
        let mut monitor = PhiMonitor::new();

        // Should succeed with good Φ
        assert!(monitor.begin_operation("test-1", 0.6).is_ok());

        // Should fail with low Φ
        assert!(monitor.begin_operation("test-2", 0.1).is_err());
        assert_eq!(monitor.stats.blocked_generations, 1);
    }

    #[test]
    fn test_full_operation_cycle() {
        let mut monitor = PhiMonitor::new();

        monitor.begin_operation("test", 0.6).unwrap();
        let analysis = monitor.end_operation(0.55, PhiMetadata::default());

        assert_eq!(analysis.pre_phi, 0.6);
        assert_eq!(analysis.post_phi, 0.55);
        assert!((analysis.delta_phi - (-0.05)).abs() < 0.001);
    }

    #[test]
    fn test_assessment() {
        assert_eq!(assess_phi_change(0.6, 0.7), PhiAssessment::Enhanced);
        assert_eq!(assess_phi_change(0.6, 0.58), PhiAssessment::Maintained);
        assert_eq!(
            assess_phi_change(0.6, 0.45),
            PhiAssessment::SignificantDecrease
        );
        assert_eq!(assess_phi_change(0.5, 0.2), PhiAssessment::BelowThreshold);
    }

    #[test]
    fn test_phi_ok_helper() {
        assert!(phi_ok_for_generation(0.5));
        assert!(phi_ok_for_generation(0.3));
        assert!(!phi_ok_for_generation(0.29));
    }

    #[test]
    fn test_trend_calculation() {
        let mut monitor = PhiMonitor::new();

        // Add measurements with rising trend
        for i in 0..5 {
            let measurement = PhiMeasurement {
                phi: 0.4 + (i as f32 * 0.05),
                timestamp: std::time::SystemTime::now(),
                context: MeasurementContext::Idle,
                metadata: PhiMetadata::default(),
            };
            monitor.record_measurement(measurement);
        }

        assert_eq!(monitor.calculate_trend(), PhiTrend::Rising);
    }

    #[test]
    fn test_stats_tracking() {
        let mut monitor = PhiMonitor::new();

        // Use delta > 0.05 to trigger Enhanced assessment
        monitor.begin_operation("test", 0.6).unwrap();
        let _ = monitor.end_operation(0.66, PhiMetadata::default());

        assert_eq!(monitor.stats.total_generations, 1);
        assert_eq!(monitor.stats.phi_enhanced, 1);
    }
}
