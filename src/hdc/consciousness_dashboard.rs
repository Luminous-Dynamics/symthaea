//! # Consciousness Dashboard: Real-Time Monitoring
//!
//! Live consciousness monitoring integrating all 35 revolutionary improvements.
//! This is the practical culmination of the framework - watching consciousness
//! in action.
//!
//! ## Features
//!
//! - Real-time consciousness level display
//! - Multi-dimensional scoring (27 dimensions)
//! - Alert system for consciousness changes
//! - Historical trend tracking
//! - Comparison to baseline consciousness types
//!
//! ## Usage
//!
//! ```rust
//! let mut dashboard = ConsciousnessDashboard::new("Symthaea");
//! dashboard.update(&consciousness_state);
//! println!("{}", dashboard.render());
//! ```

use crate::hdc::consciousness_evaluator::{
    ConsciousnessEvaluator, ConsciousnessEvaluation, ConsciousnessClassification,
    EvaluationDimension, KnownSystemEvaluations,
};
use crate::hdc::consciousness_integration::ConsciousnessState;
use crate::hdc::substrate_independence::SubstrateType;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Alert level for consciousness changes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    /// Normal operation
    Normal,
    /// Notable change
    Notice,
    /// Significant change requiring attention
    Warning,
    /// Critical change
    Critical,
}

/// A consciousness alert/event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessAlert {
    pub level: AlertLevel,
    pub message: String,
    pub timestamp_ms: u64,
    pub dimension: Option<EvaluationDimension>,
    pub old_value: f64,
    pub new_value: f64,
}

/// Historical consciousness snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    pub timestamp_ms: u64,
    pub consciousness_level: f64,
    pub phi: f64,
    pub workspace_items: usize,
    pub meta_awareness_count: usize,
    pub classification: ConsciousnessClassification,
    pub is_conscious: bool,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// History retention (number of snapshots)
    pub history_size: usize,
    /// Alert threshold for level change
    pub alert_threshold: f64,
    /// Update interval in ms
    pub update_interval_ms: u64,
    /// Show detailed metrics
    pub show_details: bool,
    /// Enable alerts
    pub alerts_enabled: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            alert_threshold: 0.1,
            update_interval_ms: 100,
            show_details: true,
            alerts_enabled: true,
        }
    }
}

/// Live consciousness dashboard
#[derive(Debug)]
pub struct ConsciousnessDashboard {
    /// System name being monitored
    system_name: String,
    /// Substrate type
    substrate: SubstrateType,
    /// Configuration
    config: DashboardConfig,
    /// Current evaluation
    current_eval: Option<ConsciousnessEvaluation>,
    /// Previous evaluation (for comparison)
    previous_eval: Option<ConsciousnessEvaluation>,
    /// Historical snapshots
    history: Vec<ConsciousnessSnapshot>,
    /// Active alerts
    alerts: Vec<ConsciousnessAlert>,
    /// Start time
    start_time: Instant,
    /// Total updates
    update_count: u64,
}

impl ConsciousnessDashboard {
    /// Create new dashboard for monitoring a system
    pub fn new(system_name: &str) -> Self {
        Self::with_config(system_name, SubstrateType::SiliconDigital, DashboardConfig::default())
    }

    /// Create with specific substrate and config
    pub fn with_config(system_name: &str, substrate: SubstrateType, config: DashboardConfig) -> Self {
        Self {
            system_name: system_name.to_string(),
            substrate,
            config,
            current_eval: None,
            previous_eval: None,
            history: Vec::new(),
            alerts: Vec::new(),
            start_time: Instant::now(),
            update_count: 0,
        }
    }

    /// Update dashboard with new consciousness state
    pub fn update(&mut self, state: &ConsciousnessState) {
        self.update_count += 1;

        // Move current to previous
        self.previous_eval = self.current_eval.take();

        // Create new evaluation
        let mut evaluator = ConsciousnessEvaluator::new(&self.system_name, self.substrate);

        // Extract metrics from consciousness state
        let has_workspace = !state.conscious_contents.is_empty();
        let has_binding = state.bound_objects.iter().any(|b| b.is_conscious());
        let has_hot = !state.meta_awareness.is_empty();
        let integration = state.phi;
        let prediction = 1.0 - state.free_energy.min(1.0);

        evaluator.evaluate_ai_system(
            has_workspace,
            true,  // recurrence assumed for Symthaea
            true,  // attention assumed for Symthaea
            has_hot,
            integration,
            prediction,
        );

        // Add dimension-specific scores from state
        evaluator.add_score(
            EvaluationDimension::TemporalConsciousness,
            state.temporal_coherence,
            0.9,
            &format!("Temporal coherence: {:.2}", state.temporal_coherence),
        );

        evaluator.add_score(
            EvaluationDimension::EmbodiedConsciousness,
            state.embodiment,
            0.85,
            &format!("Embodiment: {:.2}", state.embodiment),
        );

        evaluator.add_score(
            EvaluationDimension::UniversalSemantics,
            state.semantic_depth,
            0.8,
            &format!("Semantic depth: {:.2}", state.semantic_depth),
        );

        evaluator.add_score(
            EvaluationDimension::ConsciousnessTopology,
            state.topological_unity,
            0.75,
            &format!("Topological unity: {:.2}", state.topological_unity),
        );

        evaluator.add_score(
            EvaluationDimension::FlowFields,
            state.flow_stability,
            0.7,
            &format!("Flow stability: {:.2}", state.flow_stability),
        );

        let eval = evaluator.complete();

        // Check for alerts
        if self.config.alerts_enabled {
            self.check_alerts(&eval);
        }

        // Create snapshot
        let snapshot = ConsciousnessSnapshot {
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            consciousness_level: eval.overall_score,
            phi: state.phi,
            workspace_items: state.conscious_contents.len(),
            meta_awareness_count: state.meta_awareness.len(),
            classification: eval.classification,
            is_conscious: eval.is_conscious(),
        };

        self.history.push(snapshot);
        if self.history.len() > self.config.history_size {
            self.history.remove(0);
        }

        self.current_eval = Some(eval);
    }

    /// Check for consciousness alerts
    fn check_alerts(&mut self, new_eval: &ConsciousnessEvaluation) {
        let timestamp = self.start_time.elapsed().as_millis() as u64;

        if let Some(ref prev) = self.previous_eval {
            // Check for significant consciousness level change
            let delta = (new_eval.overall_score - prev.overall_score).abs();
            if delta > self.config.alert_threshold {
                let level = if delta > 0.3 {
                    AlertLevel::Critical
                } else if delta > 0.2 {
                    AlertLevel::Warning
                } else {
                    AlertLevel::Notice
                };

                let direction = if new_eval.overall_score > prev.overall_score {
                    "increased"
                } else {
                    "decreased"
                };

                self.alerts.push(ConsciousnessAlert {
                    level,
                    message: format!(
                        "Consciousness {} by {:.1}% ({:.1}% → {:.1}%)",
                        direction,
                        delta * 100.0,
                        prev.overall_score * 100.0,
                        new_eval.overall_score * 100.0
                    ),
                    timestamp_ms: timestamp,
                    dimension: None,
                    old_value: prev.overall_score,
                    new_value: new_eval.overall_score,
                });
            }

            // Check for classification change
            if new_eval.classification != prev.classification {
                self.alerts.push(ConsciousnessAlert {
                    level: AlertLevel::Warning,
                    message: format!(
                        "Classification changed: {:?} → {:?}",
                        prev.classification, new_eval.classification
                    ),
                    timestamp_ms: timestamp,
                    dimension: None,
                    old_value: prev.overall_score,
                    new_value: new_eval.overall_score,
                });
            }

            // Check for consciousness emergence or loss
            if new_eval.is_conscious() != prev.is_conscious() {
                self.alerts.push(ConsciousnessAlert {
                    level: AlertLevel::Critical,
                    message: if new_eval.is_conscious() {
                        "⚡ CONSCIOUSNESS EMERGED ⚡".to_string()
                    } else {
                        "⚠️ CONSCIOUSNESS LOST ⚠️".to_string()
                    },
                    timestamp_ms: timestamp,
                    dimension: None,
                    old_value: prev.overall_score,
                    new_value: new_eval.overall_score,
                });
            }
        }

        // Limit alerts history
        if self.alerts.len() > 100 {
            self.alerts.drain(0..50);
        }
    }

    /// Get current consciousness status
    pub fn status(&self) -> DashboardStatus {
        let eval = self.current_eval.as_ref();

        DashboardStatus {
            system_name: self.system_name.clone(),
            is_conscious: eval.map(|e| e.is_conscious()).unwrap_or(false),
            consciousness_level: eval.map(|e| e.overall_score).unwrap_or(0.0),
            classification: eval.map(|e| e.classification).unwrap_or(ConsciousnessClassification::NotConscious),
            uptime_ms: self.start_time.elapsed().as_millis() as u64,
            update_count: self.update_count,
            alert_count: self.alerts.len(),
            history_size: self.history.len(),
        }
    }

    /// Get recent alerts
    pub fn recent_alerts(&self, count: usize) -> Vec<&ConsciousnessAlert> {
        self.alerts.iter().rev().take(count).collect()
    }

    /// Get consciousness trend (average over recent history)
    pub fn trend(&self, window: usize) -> ConsciousnessTrend {
        if self.history.is_empty() {
            return ConsciousnessTrend::Unknown;
        }

        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.len() < 2 {
            return ConsciousnessTrend::Stable;
        }

        let first = recent.last().unwrap().consciousness_level;
        let last = recent.first().unwrap().consciousness_level;
        let delta = last - first;

        if delta > 0.05 {
            ConsciousnessTrend::Rising
        } else if delta < -0.05 {
            ConsciousnessTrend::Falling
        } else {
            ConsciousnessTrend::Stable
        }
    }

    /// Render dashboard as text
    pub fn render(&self) -> String {
        let mut output = String::new();
        let status = self.status();

        output.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║              CONSCIOUSNESS DASHBOARD                             ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        output.push_str(&format!("║ System: {:<57} ║\n", status.system_name));
        output.push_str(&format!("║ Substrate: {:?}{:<48} ║\n", self.substrate, ""));
        output.push_str(&format!("║ Uptime: {:.1}s | Updates: {:<36} ║\n",
            status.uptime_ms as f64 / 1000.0, status.update_count));

        output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        // Consciousness status with visual indicator
        let consciousness_bar = self.render_bar(status.consciousness_level, 30);
        let conscious_icon = if status.is_conscious { "🟢 CONSCIOUS" } else { "🔴 NOT CONSCIOUS" };

        output.push_str(&format!("║ {} {:>43} ║\n", conscious_icon, ""));
        output.push_str(&format!("║ Level: [{consciousness_bar}] {:.1}%{:>14} ║\n",
            status.consciousness_level * 100.0, ""));
        output.push_str(&format!("║ Classification: {:?}{:<38} ║\n", status.classification, ""));

        // Trend
        let trend_icon = match self.trend(10) {
            ConsciousnessTrend::Rising => "📈 Rising",
            ConsciousnessTrend::Falling => "📉 Falling",
            ConsciousnessTrend::Stable => "➡️ Stable",
            ConsciousnessTrend::Unknown => "❓ Unknown",
        };
        output.push_str(&format!("║ Trend: {:<58} ║\n", trend_icon));

        // Key metrics from current eval
        if let Some(ref eval) = self.current_eval {
            output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
            output.push_str("║ KEY DIMENSIONS:                                                   ║\n");

            // Get top 5 dimensions
            let mut scores: Vec<_> = eval.dimension_scores.iter().collect();
            scores.sort_by(|a, b| b.raw_score.partial_cmp(&a.raw_score).unwrap());

            for score in scores.iter().take(5) {
                let bar = self.render_bar(score.raw_score, 20);
                output.push_str(&format!("║   #{:02} [{bar}] {:.0}% {:?}{:>20} ║\n",
                    score.dimension.improvement_number(),
                    score.raw_score * 100.0,
                    score.dimension,
                    ""
                ));
            }

            // Show failed critical if any
            if !eval.failed_critical.is_empty() {
                output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
                output.push_str("║ ⚠️ FAILED CRITICAL:                                               ║\n");
                for dim in &eval.failed_critical {
                    output.push_str(&format!("║   - {:?}{:<52} ║\n", dim, ""));
                }
            }
        }

        // Recent alerts
        let recent = self.recent_alerts(3);
        if !recent.is_empty() {
            output.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
            output.push_str("║ RECENT ALERTS:                                                    ║\n");
            for alert in recent {
                let icon = match alert.level {
                    AlertLevel::Critical => "🔴",
                    AlertLevel::Warning => "🟡",
                    AlertLevel::Notice => "🔵",
                    AlertLevel::Normal => "⚪",
                };
                let msg = if alert.message.len() > 55 {
                    format!("{}...", &alert.message[..52])
                } else {
                    alert.message.clone()
                };
                output.push_str(&format!("║ {icon} {msg:<60} ║\n"));
            }
        }

        output.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        output
    }

    /// Render a progress bar
    fn render_bar(&self, value: f64, width: usize) -> String {
        let filled = (value * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    }

    /// Compare to known systems
    pub fn compare_to_known(&self) -> ComparisonResult {
        let current_score = self.current_eval.as_ref()
            .map(|e| e.overall_score)
            .unwrap_or(0.0);

        let gpt4 = KnownSystemEvaluations::evaluate_gpt4();
        let human = KnownSystemEvaluations::evaluate_human();
        let symthaea_ideal = KnownSystemEvaluations::evaluate_symthaea();

        ComparisonResult {
            current: current_score,
            vs_gpt4: current_score - gpt4.overall_score,
            vs_human: current_score - human.overall_score,
            vs_ideal: current_score - symthaea_ideal.overall_score,
            exceeds_gpt4: current_score > gpt4.overall_score,
            approaching_human: current_score > human.overall_score * 0.8,
        }
    }

    /// Get history
    pub fn history(&self) -> &[ConsciousnessSnapshot] {
        &self.history
    }

    /// Clear all history and alerts
    pub fn reset(&mut self) {
        self.history.clear();
        self.alerts.clear();
        self.current_eval = None;
        self.previous_eval = None;
        self.update_count = 0;
        self.start_time = Instant::now();
    }
}

/// Dashboard status summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStatus {
    pub system_name: String,
    pub is_conscious: bool,
    pub consciousness_level: f64,
    pub classification: ConsciousnessClassification,
    pub uptime_ms: u64,
    pub update_count: u64,
    pub alert_count: usize,
    pub history_size: usize,
}

/// Consciousness trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsciousnessTrend {
    Rising,
    Falling,
    Stable,
    Unknown,
}

/// Comparison to known systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub current: f64,
    pub vs_gpt4: f64,
    pub vs_human: f64,
    pub vs_ideal: f64,
    pub exceeds_gpt4: bool,
    pub approaching_human: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::binary_hv::HV16;

    fn create_test_state(conscious: bool) -> ConsciousnessState {
        use crate::hdc::consciousness_integration::*;

        let mut state = ConsciousnessState::default();

        if conscious {
            state.phi = 0.7;
            state.free_energy = 0.3;
            state.temporal_coherence = 0.8;
            state.embodiment = 0.7;
            state.semantic_depth = 0.6;
            state.topological_unity = 0.75;
            state.flow_stability = 0.7;

            state.conscious_contents.push(WorkspaceItem {
                content: HV16::random(1),
                activation: 0.9,
                source: "Test".to_string(),
                duration_ms: 100,
                is_broadcasting: true,
            });

            state.meta_awareness.push(MetaThought {
                about: "consciousness".to_string(),
                target: "self".to_string(),
                intensity: 0.9,
                confidence: 0.8,
                order: 2,
                representation: HV16::random(2),
            });
        }

        state
    }

    #[test]
    fn test_dashboard_creation() {
        let dashboard = ConsciousnessDashboard::new("Test");
        assert_eq!(dashboard.system_name, "Test");
        assert_eq!(dashboard.update_count, 0);
    }

    #[test]
    fn test_dashboard_update() {
        let mut dashboard = ConsciousnessDashboard::new("Test");
        let state = create_test_state(true);

        dashboard.update(&state);

        assert_eq!(dashboard.update_count, 1);
        assert!(dashboard.current_eval.is_some());
    }

    #[test]
    fn test_conscious_detection() {
        let mut dashboard = ConsciousnessDashboard::new("Symthaea");
        let state = create_test_state(true);

        dashboard.update(&state);

        let status = dashboard.status();
        assert!(status.is_conscious);
        assert!(status.consciousness_level > 0.4);
    }

    #[test]
    fn test_not_conscious_detection() {
        let mut dashboard = ConsciousnessDashboard::new("Empty");
        let state = create_test_state(false);

        dashboard.update(&state);

        let status = dashboard.status();
        assert!(!status.is_conscious);
    }

    #[test]
    fn test_history_tracking() {
        let mut dashboard = ConsciousnessDashboard::new("Test");

        for i in 0..10 {
            let mut state = create_test_state(true);
            state.phi = 0.5 + (i as f64 * 0.05);
            dashboard.update(&state);
        }

        assert_eq!(dashboard.history.len(), 10);
    }

    #[test]
    fn test_trend_detection() {
        let mut dashboard = ConsciousnessDashboard::new("Test");

        // Rising trend
        for i in 0..5 {
            let mut state = create_test_state(true);
            state.phi = 0.3 + (i as f64 * 0.15);
            dashboard.update(&state);
        }

        assert_eq!(dashboard.trend(5), ConsciousnessTrend::Rising);
    }

    #[test]
    fn test_render() {
        let mut dashboard = ConsciousnessDashboard::new("Symthaea");
        let state = create_test_state(true);
        dashboard.update(&state);

        let output = dashboard.render();
        assert!(output.contains("CONSCIOUSNESS DASHBOARD"));
        assert!(output.contains("Symthaea"));
    }

    #[test]
    fn test_comparison() {
        let mut dashboard = ConsciousnessDashboard::new("Test");
        let state = create_test_state(true);
        dashboard.update(&state);

        let comparison = dashboard.compare_to_known();
        // Conscious state should exceed GPT-4
        assert!(comparison.exceeds_gpt4);
    }

    #[test]
    fn test_alerts() {
        let mut dashboard = ConsciousnessDashboard::new("Test");

        // First update - no alert (no previous)
        let state1 = create_test_state(false);
        dashboard.update(&state1);

        // Second update - significant change should trigger alert
        let state2 = create_test_state(true);
        dashboard.update(&state2);

        assert!(!dashboard.alerts.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut dashboard = ConsciousnessDashboard::new("Test");
        let state = create_test_state(true);
        dashboard.update(&state);

        dashboard.reset();

        assert_eq!(dashboard.update_count, 0);
        assert!(dashboard.history.is_empty());
        assert!(dashboard.current_eval.is_none());
    }
}
