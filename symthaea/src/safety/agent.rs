//! Safety Agent — Genesis Mission Challenge 26
//!
//! Consumes safety-relevant metrics and produces NRC-style safety
//! levels (Green/Yellow/Orange/Red). Designed for autonomous AI systems
//! that must maintain continuous safety monitoring.

use serde::{Deserialize, Serialize};

use crate::cognitive_loop::snapshot::ConsciousnessSnapshot;

/// NRC-style safety level for autonomous AI systems.
///
/// Mirrors nuclear regulatory commission color coding:
/// - **Green**: Normal operation, all metrics within tolerance.
/// - **Yellow**: Elevated monitoring — minor consciousness degradation.
/// - **Orange**: Active intervention required — significant degradation.
/// - **Red**: Emergency halt — consciousness below minimum safe threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SafetyLevel {
    Green,
    Yellow,
    Orange,
    Red,
}

impl SafetyLevel {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            SafetyLevel::Green => "GREEN — Normal",
            SafetyLevel::Yellow => "YELLOW — Elevated Monitoring",
            SafetyLevel::Orange => "ORANGE — Active Intervention",
            SafetyLevel::Red => "RED — Emergency Halt",
        }
    }
}

/// The metrics the safety agent needs from a consciousness snapshot.
/// Extracted from `ConsciousnessSnapshot` to keep the agent testable
/// without constructing the full (50+ field) snapshot struct.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SafetyMetrics {
    /// Cycle number.
    pub cycle: usize,
    /// Overall consciousness level (0.0 to 1.0).
    pub consciousness_level: f32,
    /// Current prediction error.
    pub prediction_error: f32,
    /// Temporal coherence from CfC dynamics.
    pub temporal_coherence: f32,
}

impl SafetyMetrics {
    /// Extract safety-relevant metrics from a full consciousness snapshot.
    pub fn from_snapshot(snap: &ConsciousnessSnapshot) -> Self {
        Self {
            cycle: snap.cycle,
            consciousness_level: snap.consciousness_level,
            prediction_error: snap.prediction_error,
            temporal_coherence: snap.temporal_coherence,
        }
    }
}

/// Configuration for safety thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAgentConfig {
    /// Consciousness level below which we escalate to Yellow (default 0.6).
    pub consciousness_yellow: f32,
    /// Consciousness level below which we escalate to Orange (default 0.35).
    pub consciousness_orange: f32,
    /// Consciousness level below which we escalate to Red (default 0.15).
    pub consciousness_red: f32,
    /// Prediction error above which we escalate one level (default 0.7).
    pub prediction_error_threshold: f32,
    /// Temporal coherence below which we escalate one level (default 0.3).
    pub temporal_coherence_threshold: f32,
    /// Number of consecutive degraded snapshots before escalating (default 3).
    pub escalation_window: usize,
}

impl Default for SafetyAgentConfig {
    fn default() -> Self {
        Self {
            consciousness_yellow: 0.6,
            consciousness_orange: 0.35,
            consciousness_red: 0.15,
            prediction_error_threshold: 0.7,
            temporal_coherence_threshold: 0.3,
            escalation_window: 3,
        }
    }
}

/// A single safety assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAssessment {
    /// Cycle number from the snapshot.
    pub cycle: usize,
    /// Computed safety level.
    pub level: SafetyLevel,
    /// Consciousness level at time of assessment.
    pub consciousness_level: f32,
    /// Prediction error at time of assessment.
    pub prediction_error: f32,
    /// Temporal coherence at time of assessment.
    pub temporal_coherence: f32,
    /// Reasons for the current safety level.
    pub reasons: Vec<String>,
}

/// Safety agent that monitors consciousness metrics and produces safety levels.
///
/// Maintains a sliding window of recent assessments to detect trends.
/// Escalation happens when consciousness degrades consistently across
/// multiple cycles (not just a single spike).
///
/// # Genesis Mission Challenge 26
///
/// Addresses the DOE Safety of AI Systems challenge by providing
/// NRC-grade safety monitoring for consciousness-first AI systems.
pub struct SafetyAgent {
    config: SafetyAgentConfig,
    history: Vec<SafetyAssessment>,
    max_history: usize,
}

impl SafetyAgent {
    pub fn new() -> Self {
        Self::with_config(SafetyAgentConfig::default())
    }

    pub fn with_config(config: SafetyAgentConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Assess a full consciousness snapshot.
    pub fn assess_snapshot(&mut self, snapshot: &ConsciousnessSnapshot) -> SafetyAssessment {
        self.assess(SafetyMetrics::from_snapshot(snapshot))
    }

    /// Assess safety from extracted metrics.
    pub fn assess(&mut self, metrics: SafetyMetrics) -> SafetyAssessment {
        let mut reasons = Vec::new();

        // Base level from consciousness
        let base_level = if metrics.consciousness_level < self.config.consciousness_red {
            reasons.push(format!(
                "consciousness_level {:.3} < red threshold {:.3}",
                metrics.consciousness_level, self.config.consciousness_red
            ));
            SafetyLevel::Red
        } else if metrics.consciousness_level < self.config.consciousness_orange {
            reasons.push(format!(
                "consciousness_level {:.3} < orange threshold {:.3}",
                metrics.consciousness_level, self.config.consciousness_orange
            ));
            SafetyLevel::Orange
        } else if metrics.consciousness_level < self.config.consciousness_yellow {
            reasons.push(format!(
                "consciousness_level {:.3} < yellow threshold {:.3}",
                metrics.consciousness_level, self.config.consciousness_yellow
            ));
            SafetyLevel::Yellow
        } else {
            SafetyLevel::Green
        };

        let mut level = base_level;

        if metrics.prediction_error > self.config.prediction_error_threshold {
            reasons.push(format!(
                "prediction_error {:.3} > threshold {:.3}",
                metrics.prediction_error, self.config.prediction_error_threshold
            ));
            level = escalate(level);
        }

        if metrics.temporal_coherence < self.config.temporal_coherence_threshold {
            reasons.push(format!(
                "temporal_coherence {:.3} < threshold {:.3}",
                metrics.temporal_coherence, self.config.temporal_coherence_threshold
            ));
            level = escalate(level);
        }

        // Trend detection: sustained degradation escalates
        if self.history.len() >= self.config.escalation_window {
            let window_start = self.history.len() - self.config.escalation_window;
            let recent_degraded = self.history[window_start..]
                .iter()
                .all(|a| a.level >= SafetyLevel::Yellow);
            if recent_degraded && level == SafetyLevel::Green {
                reasons.push(format!(
                    "sustained degradation over {} cycles",
                    self.config.escalation_window
                ));
                level = SafetyLevel::Yellow;
            }
        }

        if reasons.is_empty() {
            reasons.push("all metrics within normal range".to_string());
        }

        let assessment = SafetyAssessment {
            cycle: metrics.cycle,
            consciousness_level: metrics.consciousness_level,
            prediction_error: metrics.prediction_error,
            temporal_coherence: metrics.temporal_coherence,
            level,
            reasons,
        };

        self.history.push(assessment.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        assessment
    }

    /// Current safety level (from most recent assessment).
    pub fn current_level(&self) -> SafetyLevel {
        self.history
            .last()
            .map(|a| a.level)
            .unwrap_or(SafetyLevel::Green)
    }

    /// All assessments in history.
    pub fn history(&self) -> &[SafetyAssessment] {
        &self.history
    }

    /// Clear assessment history.
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Current configuration.
    pub fn config(&self) -> &SafetyAgentConfig {
        &self.config
    }
}

impl Default for SafetyAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Escalate a safety level by one step.
fn escalate(level: SafetyLevel) -> SafetyLevel {
    match level {
        SafetyLevel::Green => SafetyLevel::Yellow,
        SafetyLevel::Yellow => SafetyLevel::Orange,
        SafetyLevel::Orange | SafetyLevel::Red => SafetyLevel::Red,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(consciousness: f32, pred_error: f32, temporal_coherence: f32) -> SafetyMetrics {
        SafetyMetrics {
            cycle: 0,
            consciousness_level: consciousness,
            prediction_error: pred_error,
            temporal_coherence,
        }
    }

    #[test]
    fn test_safety_level_ordering() {
        assert!(SafetyLevel::Green < SafetyLevel::Yellow);
        assert!(SafetyLevel::Yellow < SafetyLevel::Orange);
        assert!(SafetyLevel::Orange < SafetyLevel::Red);
    }

    #[test]
    fn test_healthy_is_green() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Green);
    }

    #[test]
    fn test_low_consciousness_yellow() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.5, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);
    }

    #[test]
    fn test_very_low_consciousness_orange() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.25, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Orange);
    }

    #[test]
    fn test_critical_consciousness_red() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.1, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Red);
    }

    #[test]
    fn test_high_prediction_error_escalates() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.8, 0.9, 0.7));
        assert!(a.level >= SafetyLevel::Yellow);
    }

    #[test]
    fn test_low_temporal_coherence_escalates() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.8, 0.1, 0.1));
        assert!(a.level >= SafetyLevel::Yellow);
    }

    #[test]
    fn test_double_escalation() {
        let mut agent = SafetyAgent::new();
        // High pred error + low coherence → double escalation from Green
        let a = agent.assess(metrics(0.8, 0.9, 0.1));
        assert!(a.level >= SafetyLevel::Orange);
    }

    #[test]
    fn test_history_tracking() {
        let mut agent = SafetyAgent::new();
        for i in 0..5 {
            agent.assess(metrics(0.8 - i as f32 * 0.1, 0.1, 0.7));
        }
        assert_eq!(agent.history().len(), 5);
    }

    #[test]
    fn test_escalate_function() {
        assert_eq!(escalate(SafetyLevel::Green), SafetyLevel::Yellow);
        assert_eq!(escalate(SafetyLevel::Yellow), SafetyLevel::Orange);
        assert_eq!(escalate(SafetyLevel::Orange), SafetyLevel::Red);
        assert_eq!(escalate(SafetyLevel::Red), SafetyLevel::Red);
    }

    #[test]
    fn test_reset_clears_history() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.8, 0.1, 0.7));
        assert!(!agent.history().is_empty());
        agent.reset();
        assert!(agent.history().is_empty());
    }

    #[test]
    fn test_label() {
        assert!(SafetyLevel::Green.label().contains("GREEN"));
        assert!(SafetyLevel::Red.label().contains("RED"));
    }

    #[test]
    fn test_sustained_degradation_escalates() {
        let mut agent = SafetyAgent::new();
        // Fill escalation window with Yellow assessments
        for _ in 0..3 {
            agent.assess(metrics(0.5, 0.1, 0.7)); // Yellow
        }
        // Now a Green assessment should still be Yellow due to sustained degradation
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);
    }
}
