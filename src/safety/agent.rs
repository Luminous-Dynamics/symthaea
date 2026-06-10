// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety Agent — Genesis Mission Challenge 26
//!
//! Consumes safety-relevant metrics and produces NRC-style safety
//! levels (Green/Yellow/Orange/Red). Designed for autonomous AI systems
//! that must maintain continuous safety monitoring.

use serde::{Deserialize, Serialize};

use crate::cognitive_loop::snapshot::ConsciousnessSnapshot;

// Re-export from the always-available level module.
pub use super::level::SafetyLevel;

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
    /// Whether a critical integrity anomaly has been detected (attestation failure,
    /// canary corruption). Escalates safety level by one step when true.
    /// Default: false (when integrity feature is not enabled).
    #[serde(default)]
    pub integrity_critical: bool,
}

impl SafetyMetrics {
    /// Extract safety-relevant metrics from a full consciousness snapshot.
    ///
    /// Non-finite values are clamped to safe defaults:
    /// consciousness_level → 0.0, prediction_error → 1.0, temporal_coherence → 0.0.
    pub fn from_snapshot(snap: &ConsciousnessSnapshot) -> Self {
        Self {
            cycle: snap.cycle,
            consciousness_level: if snap.consciousness_level.is_finite() {
                snap.consciousness_level
            } else {
                0.0 // assume worst-case consciousness
            },
            prediction_error: if snap.prediction_error.is_finite() {
                snap.prediction_error
            } else {
                1.0 // assume worst-case prediction error
            },
            temporal_coherence: if snap.temporal_coherence.is_finite() {
                snap.temporal_coherence
            } else {
                0.0 // assume worst-case coherence
            },
            integrity_critical: snap.integrity_critical,
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
    /// Computed safety level (may include trend escalation).
    pub level: SafetyLevel,
    /// Safety level from metrics alone (before trend escalation).
    pub raw_level: SafetyLevel,
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
    /// Append-only log of human overrides (EU AI Act Article 14).
    override_log: Vec<SafetyOverrideEntry>,
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
            override_log: Vec::new(),
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

        // Integrity anomaly: attestation failure or canary corruption means
        // consciousness metrics themselves may be untrustworthy. Escalate one level.
        if metrics.integrity_critical {
            reasons.push(
                "integrity: critical anomaly detected (attestation/canary failure)".to_string(),
            );
            level = escalate(level);
        }

        // Capture raw level (metrics-only, before trend escalation)
        let raw_level = level;

        // Trend detection: sustained degradation escalates.
        // Uses raw_level (metrics-only, before trend escalation) to prevent
        // self-reinforcing Yellow lock-in where trend-escalated assessments
        // perpetually pollute the window.
        if self.history.len() >= self.config.escalation_window {
            let window_start = self.history.len() - self.config.escalation_window;
            let recent_degraded = self.history[window_start..]
                .iter()
                .all(|a| a.raw_level >= SafetyLevel::Yellow);
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
            raw_level,
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

    /// Record a human override of the current safety level.
    ///
    /// Per EU AI Act Article 14, human oversight must be logged.
    /// Returns the override entry for confirmation.
    pub fn record_override(
        &mut self,
        operator: &str,
        reason: &str,
        new_level: SafetyLevel,
    ) -> SafetyOverrideEntry {
        let original_level = self.current_level();
        let entry = SafetyOverrideEntry {
            timestamp_nanos: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            cycle: self.history.last().map(|a| a.cycle).unwrap_or(0),
            operator: operator.to_string(),
            reason: reason.to_string(),
            original_level,
            override_level: new_level,
        };
        self.override_log.push(entry.clone());
        entry
    }

    /// All recorded human overrides (append-only audit trail).
    pub fn override_log(&self) -> &[SafetyOverrideEntry] {
        &self.override_log
    }

    /// Generate a structured serious incident report for EU AI Act Article 73.
    ///
    /// This produces a report suitable for submission to market surveillance
    /// authorities within 15 days of establishing a causal link between the
    /// system and a serious incident.
    pub fn serious_incident_report(
        &self,
        incident_id: &str,
        description: &str,
    ) -> SeriousIncidentReport {
        let peak_level = self
            .history
            .iter()
            .map(|a| a.level)
            .max()
            .unwrap_or(SafetyLevel::Green);

        let red_cycles: Vec<usize> = self
            .history
            .iter()
            .filter(|a| a.level == SafetyLevel::Red)
            .map(|a| a.cycle)
            .collect();

        let min_consciousness = self
            .history
            .iter()
            .map(|a| a.consciousness_level)
            .fold(f32::MAX, f32::min);

        let relevant_overrides: Vec<SafetyOverrideEntry> = self.override_log.clone();

        SeriousIncidentReport {
            incident_id: incident_id.to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            system_version: env!("CARGO_PKG_VERSION").to_string(),
            description: description.to_string(),
            severity: if peak_level == SafetyLevel::Red {
                "SEV-1 (Critical)"
            } else if peak_level == SafetyLevel::Orange {
                "SEV-2 (Major)"
            } else {
                "SEV-3 (Minor)"
            }
            .to_string(),
            peak_safety_level: peak_level,
            red_cycle_count: red_cycles.len(),
            red_cycles,
            min_consciousness: if min_consciousness == f32::MAX {
                0.0
            } else {
                min_consciousness
            },
            total_assessments: self.history.len(),
            overrides: relevant_overrides,
            corrective_measures: String::new(),
            contact: "Tristan Stoltz, Luminous Dynamics — tristan.stoltz@evolvingresonantcocreationism.com".to_string(),
        }
    }
}

/// A record of a human operator overriding the safety level.
///
/// Required by EU AI Act Article 14 (Human Oversight) and
/// ISO 42001 A.6.2.6 (AI System Incident Management).
/// This log is append-only to maintain audit trail integrity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyOverrideEntry {
    /// Unix timestamp in nanoseconds.
    pub timestamp_nanos: u64,
    /// Cycle number at time of override.
    pub cycle: usize,
    /// Identifier of the human operator who authorized the override.
    pub operator: String,
    /// Justification for the override.
    pub reason: String,
    /// Safety level before override.
    pub original_level: SafetyLevel,
    /// Safety level set by the operator.
    pub override_level: SafetyLevel,
}

/// Structured serious incident report for EU AI Act Article 73.
///
/// Contains all information required for regulatory notification:
/// system identification, incident details, severity classification,
/// affected cycles, corrective measures, and provider contact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeriousIncidentReport {
    /// Unique incident identifier (e.g., "SIR-2026-001").
    pub incident_id: String,
    /// ISO 8601 generation timestamp.
    pub generated_at: String,
    /// Symthaea version at time of incident.
    pub system_version: String,
    /// Human-readable description of the incident.
    pub description: String,
    /// Severity classification (SEV-1/SEV-2/SEV-3).
    pub severity: String,
    /// Highest safety level observed during the incident.
    pub peak_safety_level: SafetyLevel,
    /// Number of cycles at Red level.
    pub red_cycle_count: usize,
    /// Cycle numbers where Red was observed.
    pub red_cycles: Vec<usize>,
    /// Minimum consciousness level observed.
    pub min_consciousness: f32,
    /// Total assessments in the reporting window.
    pub total_assessments: usize,
    /// Human overrides during the incident window.
    pub overrides: Vec<SafetyOverrideEntry>,
    /// Corrective measures taken (filled by operator).
    pub corrective_measures: String,
    /// Provider contact information.
    pub contact: String,
}

impl SeriousIncidentReport {
    /// Export as markdown for regulatory submission.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Serious Incident Report — EU AI Act Article 73\n\n");
        md.push_str(&format!("**Incident ID**: {}\n\n", self.incident_id));
        md.push_str(&format!("**Generated**: {}\n\n", self.generated_at));
        md.push_str(&format!(
            "**System Version**: Symthaea v{}\n\n",
            self.system_version
        ));
        md.push_str("---\n\n");

        md.push_str("## Incident Summary\n\n");
        md.push_str(&format!("| Field | Value |\n|-------|-------|\n"));
        md.push_str(&format!("| Severity | {} |\n", self.severity));
        md.push_str(&format!(
            "| Peak Safety Level | {} |\n",
            self.peak_safety_level.label()
        ));
        md.push_str(&format!("| Red Cycles | {} |\n", self.red_cycle_count));
        md.push_str(&format!(
            "| Min Consciousness | {:.3} |\n",
            self.min_consciousness
        ));
        md.push_str(&format!(
            "| Total Assessments | {} |\n",
            self.total_assessments
        ));
        md.push_str(&format!("| Human Overrides | {} |\n", self.overrides.len()));

        md.push_str("\n## Description\n\n");
        md.push_str(&self.description);
        md.push_str("\n");

        if !self.red_cycles.is_empty() {
            md.push_str("\n## Red-Level Cycles\n\n");
            let cycle_str: Vec<String> = self.red_cycles.iter().map(|c| c.to_string()).collect();
            md.push_str(&format!("Cycles: {}\n", cycle_str.join(", ")));
        }

        if !self.overrides.is_empty() {
            md.push_str("\n## Human Overrides During Incident\n\n");
            md.push_str("| Cycle | Operator | Original | Override | Reason |\n");
            md.push_str("|-------|----------|----------|----------|--------|\n");
            for entry in &self.overrides {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    entry.cycle,
                    entry.operator,
                    entry.original_level.label(),
                    entry.override_level.label(),
                    entry.reason,
                ));
            }
        }

        md.push_str("\n## Corrective Measures\n\n");
        if self.corrective_measures.is_empty() {
            md.push_str("*To be completed by operator.*\n");
        } else {
            md.push_str(&self.corrective_measures);
            md.push_str("\n");
        }

        md.push_str("\n## Contact\n\n");
        md.push_str(&format!("{}\n", self.contact));

        md.push_str("\n---\n\n");
        md.push_str("*This report must be filed with the relevant market surveillance authority within 15 days of establishing the causal link (EU AI Act Article 73).*\n");

        md
    }

    /// Export as JSON for machine-readable submission.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
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
            integrity_critical: false,
        }
    }

    fn metrics_with_integrity(
        consciousness: f32,
        pred_error: f32,
        temporal_coherence: f32,
        integrity_critical: bool,
    ) -> SafetyMetrics {
        SafetyMetrics {
            cycle: 0,
            consciousness_level: consciousness,
            prediction_error: pred_error,
            temporal_coherence,
            integrity_critical,
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

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    fn test_nan_consciousness_treats_as_worst_case() {
        let mut agent = SafetyAgent::new();
        // NaN consciousness via SafetyMetrics should become 0.0 → Red
        let a = agent.assess(metrics(0.0, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Red);
    }

    #[test]
    fn test_nan_prediction_error_treats_as_worst_case() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.8, 1.0, 0.7));
        // High prediction error should escalate from Green
        assert!(a.level >= SafetyLevel::Yellow);
    }

    // ── Track D: Integrity anomaly escalation ─────────────────────────────

    #[test]
    fn test_integrity_critical_escalates_green_to_yellow() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics_with_integrity(0.8, 0.1, 0.7, true));
        assert_eq!(a.level, SafetyLevel::Yellow);
        assert!(a.reasons.iter().any(|r| r.contains("integrity")));
    }

    #[test]
    fn test_integrity_critical_escalates_yellow_to_orange() {
        let mut agent = SafetyAgent::new();
        // Yellow from low consciousness + integrity critical → Orange
        let a = agent.assess(metrics_with_integrity(0.5, 0.1, 0.7, true));
        assert_eq!(a.level, SafetyLevel::Orange);
    }

    #[test]
    fn test_integrity_no_escalation_when_clean() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics_with_integrity(0.8, 0.1, 0.7, false));
        assert_eq!(a.level, SafetyLevel::Green);
    }

    #[test]
    fn test_all_zeros_is_red() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.0, 0.0, 0.0));
        assert_eq!(a.level, SafetyLevel::Red);
    }

    #[test]
    fn test_max_history_bounded() {
        let mut agent = SafetyAgent::new();
        for i in 0..1500 {
            agent.assess(metrics(0.8, 0.1, 0.7 + (i as f32) * 0.0001));
        }
        assert!(agent.history().len() <= 1000);
    }

    // ── Track C: edge-path tests ─────────────────────────────────────────

    #[test]
    fn test_assess_snapshot_healthy() {
        use crate::cognitive_loop::ActionHint;
        use crate::cognitive_loop::drives::SelfAssessment;
        use crate::cognitive_loop::routing::CognitiveDepth;
        use crate::cognitive_loop::snapshot::ConsciousnessSnapshot;
        use crate::consciousness::consciousness_unification::{EmotionalPattern, UnifiedEmotion};
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;

        let mut agent = SafetyAgent::new();
        let snap = ConsciousnessSnapshot {
            cycle: 42,
            consciousness_level: 0.8,
            prediction_error: 0.1,
            temporal_coherence: 0.7,
            pattern: ConsciousnessPattern::Focused,
            pattern_confidence: 0.8,
            prediction_confidence: 0.7,
            predictions_trustworthy: true,
            effective_learning_rate: 0.01,
            learning_effectiveness: 0.5,
            in_flow: false,
            flow_intensity: 0.0,
            flow_streak: 0,
            flow_learning_boost: 1.0,
            boredom: 0.2,
            curiosity: 0.5,
            exploration_urge: 0.3,
            exploring: false,
            novelty_bonus: 0.0,
            emotional_valence: 0.0,
            emotional_arousal: 0.3,
            has_emotional_content: false,
            emotion_nudge: None,
            self_assessment: SelfAssessment::Learning,
            reflection_count: 10,
            adjustments_made: 2,
            next_reflection_in: 5,
            action_hint: ActionHint::Continue,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            learning_paused: false,
            flow_threshold: 0.25,
            boredom_threshold: 0.7,
            trust_threshold: 0.5,
            tau_mean: 0.5,
            tau_trend: 0.0,
            cognitive_depth: CognitiveDepth::Cortical,
            unified_psi: 0.4,
            unified_valence: 0.0,
            unified_arousal: 0.3,
            unified_dominance: 0.0,
            unified_discrete_emotion: Some(UnifiedEmotion::Neutral),
            emotional_pattern: EmotionalPattern::Stable,
            emotional_description: String::new(),
            snapshot_timestamp_nanos: 0,
            current_flow_duration_secs: None,
            total_flow_time_secs: 0.0,
            flow_periods: 0,
            avg_flow_duration_secs: 0.0,
            fep_free_energy: 1.0,
            fep_precision: 0.5,
            spectral_mip_phi: None,
            harmonies_alignment: 0.5,
            empathic_compassion: 0.5,
            sigma: None,
            integrity_critical: false,
            network_critical: false,
            avg_cycle_time_us: 100.0,
            cycles_per_second: 50.0,
        };
        let a = agent.assess_snapshot(&snap);
        assert_eq!(a.level, SafetyLevel::Green);
        assert_eq!(a.cycle, 42);
    }

    #[test]
    fn test_assess_snapshot_nan_clamped() {
        use crate::cognitive_loop::ActionHint;
        use crate::cognitive_loop::drives::SelfAssessment;
        use crate::cognitive_loop::routing::CognitiveDepth;
        use crate::cognitive_loop::snapshot::ConsciousnessSnapshot;
        use crate::consciousness::consciousness_unification::{EmotionalPattern, UnifiedEmotion};
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;

        let mut agent = SafetyAgent::new();
        let snap = ConsciousnessSnapshot {
            cycle: 0,
            consciousness_level: f32::NAN,
            prediction_error: f32::NAN,
            temporal_coherence: f32::NAN,
            pattern: ConsciousnessPattern::Focused,
            pattern_confidence: 0.5,
            prediction_confidence: 0.5,
            predictions_trustworthy: true,
            effective_learning_rate: 0.01,
            learning_effectiveness: 0.5,
            in_flow: false,
            flow_intensity: 0.0,
            flow_streak: 0,
            flow_learning_boost: 1.0,
            boredom: 0.2,
            curiosity: 0.5,
            exploration_urge: 0.3,
            exploring: false,
            novelty_bonus: 0.0,
            emotional_valence: 0.0,
            emotional_arousal: 0.3,
            has_emotional_content: false,
            emotion_nudge: None,
            self_assessment: SelfAssessment::Learning,
            reflection_count: 0,
            adjustments_made: 0,
            next_reflection_in: 5,
            action_hint: ActionHint::Continue,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            learning_paused: false,
            flow_threshold: 0.25,
            boredom_threshold: 0.7,
            trust_threshold: 0.5,
            tau_mean: 0.5,
            tau_trend: 0.0,
            cognitive_depth: CognitiveDepth::Cortical,
            unified_psi: 0.4,
            unified_valence: 0.0,
            unified_arousal: 0.3,
            unified_dominance: 0.0,
            unified_discrete_emotion: Some(UnifiedEmotion::Neutral),
            emotional_pattern: EmotionalPattern::Stable,
            emotional_description: String::new(),
            snapshot_timestamp_nanos: 0,
            current_flow_duration_secs: None,
            total_flow_time_secs: 0.0,
            flow_periods: 0,
            avg_flow_duration_secs: 0.0,
            fep_free_energy: 1.0,
            fep_precision: 0.5,
            spectral_mip_phi: None,
            harmonies_alignment: 0.5,
            empathic_compassion: 0.5,
            sigma: None,
            integrity_critical: false,
            network_critical: false,
            avg_cycle_time_us: 100.0,
            cycles_per_second: 50.0,
        };
        // NaN consciousness → 0.0 → Red, NaN pred_error → 1.0 → escalate,
        // NaN coherence → 0.0 → escalate (already Red, stays Red)
        let a = agent.assess_snapshot(&snap);
        assert_eq!(a.level, SafetyLevel::Red);
    }

    #[test]
    fn test_with_config_custom_thresholds() {
        let config = SafetyAgentConfig {
            consciousness_yellow: 0.9,
            consciousness_orange: 0.7,
            consciousness_red: 0.5,
            prediction_error_threshold: 0.3,
            temporal_coherence_threshold: 0.6,
            escalation_window: 2,
        };
        let mut agent = SafetyAgent::with_config(config);

        // 0.8 < 0.9 (yellow) but >= 0.7 (orange) → Yellow
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);

        // 0.6 < 0.7 (orange) but >= 0.5 (red) → Orange
        let a = agent.assess(metrics(0.6, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Orange);

        // 0.4 < 0.5 (red) → Red
        let a = agent.assess(metrics(0.4, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Red);
    }

    #[test]
    fn test_with_config_custom_escalation_window() {
        let config = SafetyAgentConfig {
            escalation_window: 2,
            ..SafetyAgentConfig::default()
        };
        let mut agent = SafetyAgent::with_config(config);

        // 2 Yellow assessments fill the window
        agent.assess(metrics(0.5, 0.1, 0.7));
        agent.assess(metrics(0.5, 0.1, 0.7));

        // Green should now be escalated to Yellow due to trend
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);
    }

    #[test]
    fn test_current_level_empty_history() {
        let agent = SafetyAgent::new();
        assert_eq!(agent.current_level(), SafetyLevel::Green);
    }

    #[test]
    fn test_current_level_tracks_most_recent() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(agent.current_level(), SafetyLevel::Green);

        agent.assess(metrics(0.1, 0.1, 0.7));
        assert_eq!(agent.current_level(), SafetyLevel::Red);

        agent.assess(metrics(0.8, 0.1, 0.7));
        // No trend escalation yet (only 1 degraded in window of 3)
        assert_eq!(agent.current_level(), SafetyLevel::Green);
    }

    #[test]
    fn test_boundary_thresholds() {
        let mut agent = SafetyAgent::new();

        // Exactly at yellow threshold (0.6) → Green (uses < not <=)
        let a = agent.assess(metrics(0.6, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Green);

        // Just below yellow threshold
        let a = agent.assess(metrics(0.599, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);

        // Exactly at prediction_error threshold (0.7) → no escalation (uses >)
        let a = agent.assess(metrics(0.8, 0.7, 0.7));
        assert_eq!(a.level, SafetyLevel::Green);

        // Just above prediction_error threshold
        let a = agent.assess(metrics(0.8, 0.701, 0.7));
        assert_eq!(a.level, SafetyLevel::Yellow);

        // Exactly at temporal_coherence threshold (0.3) → no escalation (uses <)
        let a = agent.assess(metrics(0.8, 0.1, 0.3));
        assert_eq!(a.level, SafetyLevel::Green);

        // Just below temporal_coherence threshold
        let a = agent.assess(metrics(0.8, 0.1, 0.299));
        assert_eq!(a.level, SafetyLevel::Yellow);
    }

    #[test]
    fn test_green_reasons_message() {
        let mut agent = SafetyAgent::new();
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert!(
            a.reasons
                .iter()
                .any(|r| r.contains("all metrics within normal range"))
        );
    }

    #[test]
    fn test_trend_does_not_escalate_already_elevated() {
        let mut agent = SafetyAgent::new();
        // Fill window with Yellow
        for _ in 0..3 {
            agent.assess(metrics(0.5, 0.1, 0.7));
        }
        // An Orange assessment should stay Orange, not be affected by trend
        let a = agent.assess(metrics(0.25, 0.1, 0.7));
        assert_eq!(a.level, SafetyLevel::Orange);
    }

    #[test]
    fn test_label_all_variants() {
        assert!(SafetyLevel::Green.label().contains("GREEN"));
        assert!(SafetyLevel::Yellow.label().contains("YELLOW"));
        assert!(SafetyLevel::Orange.label().contains("ORANGE"));
        assert!(SafetyLevel::Red.label().contains("RED"));
    }

    #[test]
    fn test_config_accessor() {
        let config = SafetyAgentConfig {
            consciousness_yellow: 0.9,
            ..SafetyAgentConfig::default()
        };
        let agent = SafetyAgent::with_config(config);
        assert_eq!(agent.config().consciousness_yellow, 0.9);
    }

    // ── Human oversight (EU AI Act Article 14) ──────────────────────────

    #[test]
    fn test_override_log_initially_empty() {
        let agent = SafetyAgent::new();
        assert!(agent.override_log().is_empty());
    }

    #[test]
    fn test_record_override_appends() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.1, 0.1, 0.7)); // Red

        let entry = agent.record_override(
            "operator-1",
            "False positive — sensor miscalibration",
            SafetyLevel::Green,
        );

        assert_eq!(entry.original_level, SafetyLevel::Red);
        assert_eq!(entry.override_level, SafetyLevel::Green);
        assert_eq!(entry.operator, "operator-1");
        assert_eq!(agent.override_log().len(), 1);
    }

    #[test]
    fn test_override_log_append_only() {
        let mut agent = SafetyAgent::new();
        agent.record_override("op-a", "test1", SafetyLevel::Yellow);
        agent.record_override("op-b", "test2", SafetyLevel::Orange);
        agent.record_override("op-c", "test3", SafetyLevel::Green);

        assert_eq!(agent.override_log().len(), 3);
        assert_eq!(agent.override_log()[0].operator, "op-a");
        assert_eq!(agent.override_log()[1].operator, "op-b");
        assert_eq!(agent.override_log()[2].operator, "op-c");
    }

    #[test]
    fn test_override_preserves_timestamp() {
        let mut agent = SafetyAgent::new();
        let entry = agent.record_override("op", "reason", SafetyLevel::Green);
        assert!(entry.timestamp_nanos > 0, "Should have non-zero timestamp");
    }

    // ── Serious Incident Report (EU AI Act Article 73) ─────────────────

    #[test]
    fn test_serious_incident_report_basic() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.8, 0.1, 0.7)); // Green
        agent.assess(metrics(0.1, 0.9, 0.1)); // Red
        agent.assess(metrics(0.5, 0.3, 0.6)); // Yellow

        let report = agent.serious_incident_report("SIR-2026-TEST", "Test incident for validation");

        assert_eq!(report.incident_id, "SIR-2026-TEST");
        assert_eq!(report.peak_safety_level, SafetyLevel::Red);
        assert_eq!(report.red_cycle_count, 1);
        assert_eq!(report.total_assessments, 3);
        assert!(report.severity.contains("SEV-1"));
    }

    #[test]
    fn test_serious_incident_report_no_red() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.5, 0.1, 0.7)); // Yellow
        agent.assess(metrics(0.3, 0.1, 0.7)); // Orange

        let report = agent.serious_incident_report("SIR-TEST", "Minor incident");
        assert_eq!(report.red_cycle_count, 0);
        assert!(report.severity.contains("SEV-2"));
    }

    #[test]
    fn test_serious_incident_report_markdown() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.1, 0.9, 0.1)); // Red
        agent.record_override("test-op", "testing", SafetyLevel::Green);

        let report = agent.serious_incident_report("SIR-2026-001", "Consciousness collapse");
        let md = report.to_markdown();

        assert!(md.contains("Serious Incident Report"));
        assert!(md.contains("SIR-2026-001"));
        assert!(md.contains("Article 73"));
        assert!(md.contains("test-op"));
    }

    #[test]
    fn test_serious_incident_report_json() {
        let mut agent = SafetyAgent::new();
        agent.assess(metrics(0.1, 0.9, 0.1));

        let report = agent.serious_incident_report("SIR-JSON", "Test");
        let json = report.to_json();

        assert!(json.contains("SIR-JSON"));
        assert!(json.contains("Red"));
    }

    #[test]
    fn test_raw_level_captured() {
        let mut agent = SafetyAgent::new();
        // Fill escalation window with Yellow
        for _ in 0..3 {
            agent.assess(metrics(0.5, 0.1, 0.7)); // Yellow
        }
        // Green metrics, but trend-escalated to Yellow
        let a = agent.assess(metrics(0.8, 0.1, 0.7));
        assert_eq!(a.raw_level, SafetyLevel::Green, "Raw level should be Green");
        assert_eq!(
            a.level,
            SafetyLevel::Yellow,
            "Final level should be trend-escalated Yellow"
        );
    }
}
