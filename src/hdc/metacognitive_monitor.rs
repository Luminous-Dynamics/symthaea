//! Metacognitive Monitoring System for Consciousness-Aware Computing
//!
//! This module implements a metacognitive layer that monitors, evaluates, and
//! regulates the cognitive processes of the consciousness system. It provides
//! self-awareness of processing quality, uncertainty, and integration levels.
//!
//! # Key Concepts
//!
//! ## Metacognition
//! "Thinking about thinking" - the ability to monitor and control cognitive processes.
//! This includes:
//! - **Confidence estimation**: How certain are we about our outputs?
//! - **Uncertainty tracking**: What don't we know?
//! - **Error detection**: When are we making mistakes?
//! - **Resource allocation**: Where should attention be focused?
//!
//! ## Φ-Based Monitoring
//! Integrated Information (Φ) serves as a key metric for consciousness quality.
//! We track Φ over time and use it to guide processing decisions.
//!
//! # Scientific Foundation
//!
//! Based on research in:
//! - Metacognitive theories (Flavell, Nelson & Narens)
//! - Predictive processing and active inference
//! - Higher-order theories of consciousness (HOT)
//! - Uncertainty quantification in neural networks
//!
//! # Example
//!
//! ```ignore
//! use symthaea::hdc::metacognitive_monitor::{MetacognitiveMonitor, CognitiveEvent};
//!
//! let mut monitor = MetacognitiveMonitor::new();
//!
//! // Record processing events
//! monitor.record_event(CognitiveEvent::Processing { phi: 0.85, latency_ms: 42 });
//! monitor.record_event(CognitiveEvent::Uncertainty { source: "visual", level: 0.3 });
//!
//! // Get metacognitive assessment
//! let assessment = monitor.assess_current_state();
//! println!("Confidence: {:.2}, Coherence: {:.2}", assessment.confidence, assessment.coherence);
//! ```

use crate::hdc::real_hv::RealHV as ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Types of cognitive events that can be monitored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveEvent {
    /// Processing completed with associated Φ and latency
    Processing {
        phi: f32,
        latency_ms: u64,
    },
    /// Uncertainty detected in a processing stream
    Uncertainty {
        source: String,
        level: f32,
    },
    /// Error or anomaly detected
    Error {
        error_type: ErrorType,
        severity: f32,
    },
    /// Attention shift between processing streams
    AttentionShift {
        from: String,
        to: String,
        strength: f32,
    },
    /// Binding operation completed
    Binding {
        success: bool,
        coherence: f32,
    },
    /// Memory operation (encoding or retrieval)
    Memory {
        operation: MemoryOperation,
        confidence: f32,
    },
    /// Integration across modalities
    Integration {
        modalities: Vec<String>,
        phi_contribution: f32,
    },
}

/// Types of errors that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErrorType {
    /// Binding failure (couldn't integrate information)
    BindingFailure,
    /// Coherence drop (system becoming fragmented)
    CoherenceDrop,
    /// Timeout (processing took too long)
    Timeout,
    /// Overflow (too much information at once)
    Overflow,
    /// Conflict (contradictory information)
    Conflict,
    /// Drift (gradual degradation)
    Drift,
}

/// Types of memory operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MemoryOperation {
    Encode,
    Retrieve,
    Consolidate,
    Forget,
}

/// Configuration for the metacognitive monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveConfig {
    /// Maximum number of events to keep in history
    pub history_size: usize,
    /// Minimum Φ threshold for healthy processing
    pub phi_threshold: f32,
    /// Maximum acceptable uncertainty
    pub uncertainty_threshold: f32,
    /// Confidence decay rate (per timestep)
    pub confidence_decay: f32,
    /// Error severity weight for assessments
    pub error_weight: f32,
    /// Enable predictive monitoring
    pub enable_prediction: bool,
    /// Time window for trend analysis (in events)
    pub trend_window: usize,
}

impl Default for MetacognitiveConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            phi_threshold: 0.4,
            uncertainty_threshold: 0.5,
            confidence_decay: 0.01,
            error_weight: 2.0,
            enable_prediction: true,
            trend_window: 50,
        }
    }
}

/// Current metacognitive assessment of system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveAssessment {
    /// Overall confidence in current processing
    pub confidence: f32,
    /// Overall coherence (integration quality)
    pub coherence: f32,
    /// Current average Φ level
    pub phi_level: f32,
    /// Current uncertainty level
    pub uncertainty: f32,
    /// Error rate (recent errors / total events)
    pub error_rate: f32,
    /// Processing efficiency (useful work / total work)
    pub efficiency: f32,
    /// Predicted next Φ level
    pub predicted_phi: Option<f32>,
    /// Recommended actions
    pub recommendations: Vec<MetacognitiveRecommendation>,
    /// Timestamp of assessment
    pub timestamp: u64,
}

/// Recommendations from metacognitive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetacognitiveRecommendation {
    /// Increase integration effort
    IncreaseIntegration { target_modality: Option<String> },
    /// Reduce processing load
    ReduceLoad { suggested_reduction: f32 },
    /// Focus attention on specific stream
    FocusAttention { target: String, reason: String },
    /// Trigger memory consolidation
    ConsolidateMemory,
    /// Reset or restart processing
    ResetProcessing { severity: f32 },
    /// Request external input/clarification
    RequestClarification { topic: String },
    /// No action needed
    NoAction,
}

/// Historical statistics for trend analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HistoricalStats {
    pub phi_mean: f32,
    pub phi_std: f32,
    pub phi_trend: f32, // Positive = improving, negative = declining
    pub confidence_mean: f32,
    pub error_count: usize,
    pub processing_count: usize,
}

/// Metacognitive monitoring system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitiveMonitor {
    /// Configuration
    config: MetacognitiveConfig,
    /// Event history
    history: VecDeque<TimestampedEvent>,
    /// Current running statistics
    running_stats: RunningStats,
    /// Last assessment
    last_assessment: Option<MetacognitiveAssessment>,
    /// Internal state vector (hyperdimensional representation of metacognitive state)
    state_vector: Option<ContinuousHV>,
    /// Dimension for state vectors
    dimension: usize,
    /// Monotonic timestamp counter
    timestamp: u64,
}

/// Event with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TimestampedEvent {
    event: CognitiveEvent,
    timestamp: u64,
}

/// Running statistics for efficient computation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunningStats {
    phi_sum: f32,
    phi_sum_sq: f32,
    phi_count: usize,
    uncertainty_sum: f32,
    uncertainty_count: usize,
    error_count: usize,
    total_events: usize,
    confidence: f32,
}

impl MetacognitiveMonitor {
    /// Create a new metacognitive monitor with default configuration
    pub fn new() -> Self {
        Self::with_config(MetacognitiveConfig::default())
    }

    /// Create a new metacognitive monitor with custom configuration
    pub fn with_config(config: MetacognitiveConfig) -> Self {
        Self {
            config,
            history: VecDeque::with_capacity(1000),
            running_stats: RunningStats {
                confidence: 0.5, // Start with neutral confidence
                ..Default::default()
            },
            last_assessment: None,
            state_vector: None,
            dimension: super::HDC_DIMENSION,
            timestamp: 0,
        }
    }

    /// Record a cognitive event
    pub fn record_event(&mut self, event: CognitiveEvent) {
        self.timestamp += 1;

        // Update running statistics based on event type
        self.update_stats(&event);

        // Store in history
        let timestamped = TimestampedEvent {
            event,
            timestamp: self.timestamp,
        };

        self.history.push_back(timestamped);

        // Trim history if needed
        while self.history.len() > self.config.history_size {
            if let Some(old) = self.history.pop_front() {
                self.remove_from_stats(&old.event);
            }
        }

        // Update state vector
        self.update_state_vector();
    }

    /// Update running statistics with a new event
    fn update_stats(&mut self, event: &CognitiveEvent) {
        self.running_stats.total_events += 1;

        match event {
            CognitiveEvent::Processing { phi, latency_ms: _ } => {
                self.running_stats.phi_sum += phi;
                self.running_stats.phi_sum_sq += phi * phi;
                self.running_stats.phi_count += 1;

                // Update confidence based on Φ
                if *phi > self.config.phi_threshold {
                    self.running_stats.confidence = (self.running_stats.confidence + 0.1).min(1.0);
                } else {
                    self.running_stats.confidence = (self.running_stats.confidence - 0.05).max(0.0);
                }
            }
            CognitiveEvent::Uncertainty { level, .. } => {
                self.running_stats.uncertainty_sum += level;
                self.running_stats.uncertainty_count += 1;

                // High uncertainty reduces confidence
                if *level > self.config.uncertainty_threshold {
                    self.running_stats.confidence *= 1.0 - self.config.confidence_decay;
                }
            }
            CognitiveEvent::Error { severity, .. } => {
                self.running_stats.error_count += 1;

                // Errors significantly reduce confidence
                self.running_stats.confidence *= 1.0 - severity * self.config.error_weight * 0.1;
                self.running_stats.confidence = self.running_stats.confidence.max(0.0);
            }
            CognitiveEvent::Binding { success, coherence } => {
                if *success && *coherence > 0.5 {
                    self.running_stats.confidence = (self.running_stats.confidence + 0.05).min(1.0);
                } else if !success {
                    self.running_stats.confidence *= 0.95;
                }
            }
            CognitiveEvent::Memory { confidence, .. } => {
                // Memory operations affect overall confidence
                self.running_stats.confidence =
                    self.running_stats.confidence * 0.9 + confidence * 0.1;
            }
            CognitiveEvent::Integration { phi_contribution, .. } => {
                self.running_stats.phi_sum += phi_contribution;
                self.running_stats.phi_count += 1;
            }
            CognitiveEvent::AttentionShift { .. } => {
                // Attention shifts are noted but don't directly affect stats
            }
        }
    }

    /// Remove an event from running statistics (when evicted from history)
    fn remove_from_stats(&mut self, event: &CognitiveEvent) {
        self.running_stats.total_events = self.running_stats.total_events.saturating_sub(1);

        match event {
            CognitiveEvent::Processing { phi, .. } => {
                self.running_stats.phi_sum -= phi;
                self.running_stats.phi_sum_sq -= phi * phi;
                self.running_stats.phi_count = self.running_stats.phi_count.saturating_sub(1);
            }
            CognitiveEvent::Uncertainty { level, .. } => {
                self.running_stats.uncertainty_sum -= level;
                self.running_stats.uncertainty_count = self.running_stats.uncertainty_count.saturating_sub(1);
            }
            CognitiveEvent::Error { .. } => {
                self.running_stats.error_count = self.running_stats.error_count.saturating_sub(1);
            }
            _ => {}
        }
    }

    /// Update the hyperdimensional state vector
    fn update_state_vector(&mut self) {
        // Create a representation of current metacognitive state
        let phi_level = self.get_phi_level();
        let uncertainty = self.get_uncertainty_level();
        let error_rate = self.get_error_rate();
        let confidence = self.running_stats.confidence;

        // Encode these as components of a state vector
        let seed = (phi_level * 1000.0) as u64
            + ((uncertainty * 1000.0) as u64) * 1000
            + ((confidence * 1000.0) as u64) * 1000000;

        let base = ContinuousHV::random(self.dimension, seed);

        // Scale by current state metrics
        let scale = (phi_level + confidence) / 2.0 * (1.0 - error_rate);
        let new_state = base.scale(scale);

        // Exponential moving average with previous state
        self.state_vector = match &self.state_vector {
            Some(prev) => {
                let alpha = 0.1; // Learning rate for state updates
                let weighted_prev = prev.scale(1.0 - alpha);
                let weighted_new = new_state.scale(alpha);
                Some(weighted_prev.add(&weighted_new))
            }
            None => Some(new_state),
        };
    }

    /// Get current Φ level
    fn get_phi_level(&self) -> f32 {
        if self.running_stats.phi_count == 0 {
            return 0.5; // Default
        }
        self.running_stats.phi_sum / self.running_stats.phi_count as f32
    }

    /// Get current uncertainty level
    fn get_uncertainty_level(&self) -> f32 {
        if self.running_stats.uncertainty_count == 0 {
            return 0.3; // Default moderate uncertainty
        }
        self.running_stats.uncertainty_sum / self.running_stats.uncertainty_count as f32
    }

    /// Get current error rate
    fn get_error_rate(&self) -> f32 {
        if self.running_stats.total_events == 0 {
            return 0.0;
        }
        self.running_stats.error_count as f32 / self.running_stats.total_events as f32
    }

    /// Assess current metacognitive state
    pub fn assess_current_state(&mut self) -> MetacognitiveAssessment {
        let phi_level = self.get_phi_level();
        let uncertainty = self.get_uncertainty_level();
        let error_rate = self.get_error_rate();
        let confidence = self.running_stats.confidence;

        // Calculate coherence from Φ and uncertainty
        let coherence = (phi_level * (1.0 - uncertainty)).max(0.0).min(1.0);

        // Calculate efficiency
        let efficiency = if self.running_stats.total_events > 0 {
            (self.running_stats.phi_count as f32 / self.running_stats.total_events as f32)
                * (1.0 - error_rate)
        } else {
            0.5
        };

        // Predict next Φ if enabled
        let predicted_phi = if self.config.enable_prediction {
            self.predict_phi()
        } else {
            None
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            phi_level, uncertainty, error_rate, confidence, coherence
        );

        let assessment = MetacognitiveAssessment {
            confidence,
            coherence,
            phi_level,
            uncertainty,
            error_rate,
            efficiency,
            predicted_phi,
            recommendations,
            timestamp: self.timestamp,
        };

        self.last_assessment = Some(assessment.clone());
        assessment
    }

    /// Predict next Φ level based on trend
    fn predict_phi(&self) -> Option<f32> {
        let window = self.config.trend_window.min(self.history.len());
        if window < 10 {
            return None; // Not enough data
        }

        // Collect recent Φ values (rev to get recent, then reverse to get chronological order)
        let mut phi_values: Vec<f32> = self.history
            .iter()
            .rev()
            .take(window)
            .filter_map(|e| match &e.event {
                CognitiveEvent::Processing { phi, .. } => Some(*phi),
                CognitiveEvent::Integration { phi_contribution, .. } => Some(*phi_contribution),
                _ => None,
            })
            .collect();

        if phi_values.len() < 5 {
            return None;
        }

        // Reverse to get chronological order (oldest first) for correct trend prediction
        phi_values.reverse();

        // Simple linear trend estimation
        let n = phi_values.len() as f32;
        let sum_x: f32 = (0..phi_values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = phi_values.iter().sum();
        let sum_xy: f32 = phi_values.iter().enumerate().map(|(i, y)| i as f32 * y).sum();
        let sum_xx: f32 = (0..phi_values.len()).map(|i| (i * i) as f32).sum();

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return Some(sum_y / n); // Return mean if no trend
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict next value
        let predicted = intercept + slope * (phi_values.len() as f32);
        Some(predicted.clamp(0.0, 1.0))
    }

    /// Generate metacognitive recommendations based on current state
    fn generate_recommendations(
        &self,
        phi_level: f32,
        uncertainty: f32,
        error_rate: f32,
        confidence: f32,
        coherence: f32,
    ) -> Vec<MetacognitiveRecommendation> {
        let mut recommendations = Vec::new();

        // Low Φ: need more integration
        if phi_level < self.config.phi_threshold {
            recommendations.push(MetacognitiveRecommendation::IncreaseIntegration {
                target_modality: None,
            });
        }

        // High uncertainty: request clarification
        if uncertainty > self.config.uncertainty_threshold {
            recommendations.push(MetacognitiveRecommendation::RequestClarification {
                topic: "high_uncertainty_source".to_string(),
            });
        }

        // High error rate: consider reset
        if error_rate > 0.3 {
            recommendations.push(MetacognitiveRecommendation::ResetProcessing {
                severity: error_rate,
            });
        } else if error_rate > 0.15 {
            recommendations.push(MetacognitiveRecommendation::ReduceLoad {
                suggested_reduction: error_rate * 2.0,
            });
        }

        // Low coherence but high Φ: focus attention
        if coherence < 0.4 && phi_level > 0.5 {
            recommendations.push(MetacognitiveRecommendation::FocusAttention {
                target: "integration_bottleneck".to_string(),
                reason: "High Φ but low coherence indicates fragmentation".to_string(),
            });
        }

        // Stable high performance: consolidate
        if confidence > 0.8 && phi_level > 0.6 && error_rate < 0.05 {
            recommendations.push(MetacognitiveRecommendation::ConsolidateMemory);
        }

        // If nothing to recommend
        if recommendations.is_empty() {
            recommendations.push(MetacognitiveRecommendation::NoAction);
        }

        recommendations
    }

    /// Get historical statistics for analysis
    pub fn get_historical_stats(&self) -> HistoricalStats {
        let phi_mean = self.get_phi_level();
        let phi_count = self.running_stats.phi_count;

        let phi_std = if phi_count > 1 {
            let variance = (self.running_stats.phi_sum_sq / phi_count as f32)
                - (phi_mean * phi_mean);
            variance.abs().sqrt()
        } else {
            0.0
        };

        // Calculate trend from recent history
        let phi_trend = self.predict_phi()
            .map(|pred| pred - phi_mean)
            .unwrap_or(0.0);

        HistoricalStats {
            phi_mean,
            phi_std,
            phi_trend,
            confidence_mean: self.running_stats.confidence,
            error_count: self.running_stats.error_count,
            processing_count: self.running_stats.phi_count,
        }
    }

    /// Get the current metacognitive state vector
    pub fn get_state_vector(&self) -> Option<&ContinuousHV> {
        self.state_vector.as_ref()
    }

    /// Compare similarity between two metacognitive states
    pub fn compare_states(&self, other: &MetacognitiveMonitor) -> f32 {
        match (&self.state_vector, &other.state_vector) {
            (Some(a), Some(b)) => a.similarity(b),
            _ => 0.0,
        }
    }

    /// Reset the monitor (clear history and stats)
    pub fn reset(&mut self) {
        self.history.clear();
        self.running_stats = RunningStats {
            confidence: 0.5,
            ..Default::default()
        };
        self.last_assessment = None;
        self.state_vector = None;
        self.timestamp = 0;
    }

    /// Get last assessment without recomputing
    pub fn last_assessment(&self) -> Option<&MetacognitiveAssessment> {
        self.last_assessment.as_ref()
    }

    /// Get number of events in history
    pub fn event_count(&self) -> usize {
        self.history.len()
    }

    /// Get current timestamp
    pub fn current_timestamp(&self) -> u64 {
        self.timestamp
    }
}

impl Default for MetacognitiveMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Metacognitive controller that can take actions based on assessments
#[derive(Debug, Clone)]
pub struct MetacognitiveController {
    monitor: MetacognitiveMonitor,
    /// Whether to automatically apply recommendations
    auto_apply: bool,
    /// Callback for recommendations (stored as action history)
    action_history: Vec<(u64, MetacognitiveRecommendation)>,
}

impl MetacognitiveController {
    pub fn new(monitor: MetacognitiveMonitor) -> Self {
        Self {
            monitor,
            auto_apply: false,
            action_history: Vec::new(),
        }
    }

    /// Enable automatic application of recommendations
    pub fn enable_auto_apply(&mut self) {
        self.auto_apply = true;
    }

    /// Record an event and optionally apply recommendations
    pub fn process_event(&mut self, event: CognitiveEvent) -> Option<Vec<MetacognitiveRecommendation>> {
        self.monitor.record_event(event);

        if self.auto_apply {
            let assessment = self.monitor.assess_current_state();
            let recommendations = assessment.recommendations.clone();

            // Store recommendations in action history
            for rec in &recommendations {
                self.action_history.push((self.monitor.current_timestamp(), rec.clone()));
            }

            // Trim action history
            while self.action_history.len() > 1000 {
                self.action_history.remove(0);
            }

            Some(recommendations)
        } else {
            None
        }
    }

    /// Get the underlying monitor
    pub fn monitor(&self) -> &MetacognitiveMonitor {
        &self.monitor
    }

    /// Get mutable access to the monitor
    pub fn monitor_mut(&mut self) -> &mut MetacognitiveMonitor {
        &mut self.monitor
    }

    /// Get action history
    pub fn action_history(&self) -> &[(u64, MetacognitiveRecommendation)] {
        &self.action_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_monitoring() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record some processing events
        for i in 0..10 {
            monitor.record_event(CognitiveEvent::Processing {
                phi: 0.5 + (i as f32) * 0.03,
                latency_ms: 10 + i * 2,
            });
        }

        let assessment = monitor.assess_current_state();

        assert!(assessment.phi_level > 0.5, "Phi should be above baseline");
        assert!(assessment.confidence > 0.0, "Confidence should be positive");
        assert_eq!(assessment.error_rate, 0.0, "No errors recorded");
    }

    #[test]
    fn test_error_reduces_confidence() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record successful processing
        for _ in 0..5 {
            monitor.record_event(CognitiveEvent::Processing { phi: 0.7, latency_ms: 10 });
        }

        let before = monitor.assess_current_state();

        // Record an error
        monitor.record_event(CognitiveEvent::Error {
            error_type: ErrorType::BindingFailure,
            severity: 0.8,
        });

        let after = monitor.assess_current_state();

        assert!(after.confidence < before.confidence, "Error should reduce confidence");
        assert!(after.error_rate > 0.0, "Error rate should be positive");
    }

    #[test]
    fn test_uncertainty_tracking() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record uncertainty events
        monitor.record_event(CognitiveEvent::Uncertainty {
            source: "visual".to_string(),
            level: 0.8,
        });
        monitor.record_event(CognitiveEvent::Uncertainty {
            source: "auditory".to_string(),
            level: 0.6,
        });

        let assessment = monitor.assess_current_state();

        assert!(assessment.uncertainty > 0.5, "High uncertainty should be reflected");
    }

    #[test]
    fn test_recommendations_generation() {
        let mut monitor = MetacognitiveMonitor::new();

        // Create a low-Φ state
        for _ in 0..10 {
            monitor.record_event(CognitiveEvent::Processing { phi: 0.2, latency_ms: 10 });
        }

        let assessment = monitor.assess_current_state();

        // Should recommend increasing integration
        let has_integration_rec = assessment.recommendations.iter().any(|r| {
            matches!(r, MetacognitiveRecommendation::IncreaseIntegration { .. })
        });

        assert!(has_integration_rec, "Low Φ should trigger integration recommendation");
    }

    #[test]
    fn test_phi_prediction() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record increasing Φ values
        for i in 0..20 {
            monitor.record_event(CognitiveEvent::Processing {
                phi: 0.3 + (i as f32) * 0.02,
                latency_ms: 10,
            });
        }

        let assessment = monitor.assess_current_state();

        // Should predict higher Φ due to upward trend
        if let Some(predicted) = assessment.predicted_phi {
            assert!(predicted >= assessment.phi_level,
                "Upward trend should predict higher Φ: predicted={}, current={}",
                predicted, assessment.phi_level);
        }
    }

    #[test]
    fn test_state_vector_updates() {
        let mut monitor = MetacognitiveMonitor::new();

        assert!(monitor.get_state_vector().is_none(), "No state vector initially");

        monitor.record_event(CognitiveEvent::Processing { phi: 0.6, latency_ms: 10 });

        assert!(monitor.get_state_vector().is_some(), "State vector should exist after event");
    }

    #[test]
    fn test_controller_auto_apply() {
        let monitor = MetacognitiveMonitor::new();
        let mut controller = MetacognitiveController::new(monitor);
        controller.enable_auto_apply();

        // Process events
        for _ in 0..5 {
            let recs = controller.process_event(CognitiveEvent::Processing {
                phi: 0.3,
                latency_ms: 10
            });
            assert!(recs.is_some(), "Auto-apply should return recommendations");
        }

        assert!(!controller.action_history().is_empty(), "Action history should be populated");
    }

    #[test]
    fn test_reset() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record some events
        for _ in 0..10 {
            monitor.record_event(CognitiveEvent::Processing { phi: 0.6, latency_ms: 10 });
        }

        assert!(monitor.event_count() > 0, "Should have events");

        monitor.reset();

        assert_eq!(monitor.event_count(), 0, "Should be empty after reset");
        assert!(monitor.last_assessment().is_none(), "No assessment after reset");
    }

    #[test]
    fn test_memory_operations() {
        let mut monitor = MetacognitiveMonitor::new();

        monitor.record_event(CognitiveEvent::Memory {
            operation: MemoryOperation::Encode,
            confidence: 0.9,
        });

        monitor.record_event(CognitiveEvent::Memory {
            operation: MemoryOperation::Retrieve,
            confidence: 0.7,
        });

        let assessment = monitor.assess_current_state();

        // Memory operations should contribute to confidence
        assert!(assessment.confidence > 0.0, "Memory ops should maintain confidence");
    }

    #[test]
    fn test_historical_stats() {
        let mut monitor = MetacognitiveMonitor::new();

        // Record varied Φ values
        let phi_values = vec![0.4, 0.5, 0.6, 0.5, 0.55, 0.45, 0.5, 0.52];
        for phi in phi_values {
            monitor.record_event(CognitiveEvent::Processing { phi, latency_ms: 10 });
        }

        let stats = monitor.get_historical_stats();

        assert!(stats.phi_mean > 0.4 && stats.phi_mean < 0.6,
            "Mean should be around 0.5, got {}", stats.phi_mean);
        assert!(stats.phi_std > 0.0, "Should have some variance");
        assert_eq!(stats.error_count, 0, "No errors");
        assert_eq!(stats.processing_count, 8, "8 processing events");
    }
}
