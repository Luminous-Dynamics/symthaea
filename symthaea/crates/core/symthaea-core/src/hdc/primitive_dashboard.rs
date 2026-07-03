// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Dashboard: Real-Time Primitive Usage Monitoring
//!
//! Live monitoring of primitive activations across the consciousness system.
//! Tracks which ontological primitives are being used, their success rates,
//! and provides insights for adaptive primitive selection.
//!
//! ## Features
//!
//! - Real-time primitive activation tracking
//! - Per-tier usage statistics
//! - Success/failure rate tracking
//! - Integration with AdaptivePrimitiveSelector
//! - Primitive chain visualization
//!
//! ## Usage
//!
//! ```rust,ignore
//! let mut dashboard = PrimitiveDashboard::new();
//! dashboard.record_activation("ATTEND", true);
//! dashboard.record_activation("REMEMBER", true);
//! println!("{}", dashboard.summary());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::primitive_system::{Primitive, PrimitiveTier};

/// A recorded primitive activation event
#[derive(Debug, Clone)]
pub struct PrimitiveActivation {
    /// Name of the activated primitive
    pub primitive_name: String,
    /// Tier of the primitive
    pub tier: PrimitiveTier,
    /// Whether the activation was successful
    pub success: bool,
    /// Duration of the activation (if tracked)
    pub duration: Option<Duration>,
    /// Timestamp of activation
    pub timestamp: Instant,
    /// Context tag (e.g., "voice", "reasoning", "memory")
    pub context: String,
}

/// Statistics for a single primitive
#[derive(Debug, Clone, Default)]
pub struct PrimitiveStats {
    /// Total activation count
    pub activations: usize,
    /// Successful activations
    pub successes: usize,
    /// Average duration in milliseconds
    pub avg_duration_ms: f64,
    /// Peak activations per minute
    pub peak_rate: f64,
    /// Last activation time
    pub last_activation: Option<Instant>,
}

impl PrimitiveStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.activations == 0 {
            0.0
        } else {
            self.successes as f64 / self.activations as f64
        }
    }

    /// Record an activation
    pub fn record(&mut self, success: bool, duration: Option<Duration>) {
        self.activations += 1;
        if success {
            self.successes += 1;
        }
        if let Some(d) = duration {
            let ms = d.as_secs_f64() * 1000.0;
            // Running average
            self.avg_duration_ms = (self.avg_duration_ms * (self.activations - 1) as f64 + ms)
                / self.activations as f64;
        }
        self.last_activation = Some(Instant::now());
    }
}

/// Per-tier statistics
#[derive(Debug, Clone, Default)]
pub struct TierStats {
    /// Total activations in this tier
    pub activations: usize,
    /// Success rate
    pub successes: usize,
    /// Most used primitive in tier
    pub most_used: Option<String>,
    /// Least used primitive in tier
    pub least_used: Option<String>,
}

/// Primitive usage dashboard for real-time monitoring
#[derive(Debug)]
pub struct PrimitiveDashboard {
    /// Per-primitive statistics
    stats: HashMap<String, PrimitiveStats>,
    /// Per-tier statistics
    tier_stats: HashMap<PrimitiveTier, TierStats>,
    /// Recent activation history (ring buffer)
    history: Vec<PrimitiveActivation>,
    /// Maximum history size
    history_size: usize,
    /// Dashboard creation time
    start_time: Instant,
    /// Total activations
    total_activations: usize,
    /// Current context (for grouping)
    current_context: String,
}

impl PrimitiveDashboard {
    /// Create a new primitive dashboard
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            tier_stats: HashMap::new(),
            history: Vec::new(),
            history_size: 1000,
            start_time: Instant::now(),
            total_activations: 0,
            current_context: "general".to_string(),
        }
    }

    /// Create with custom history size
    pub fn with_history_size(history_size: usize) -> Self {
        let mut dashboard = Self::new();
        dashboard.history_size = history_size;
        dashboard
    }

    /// Set the current context for subsequent activations
    pub fn set_context(&mut self, context: &str) {
        self.current_context = context.to_string();
    }

    /// Record a primitive activation
    pub fn record_activation(&mut self, primitive_name: &str, tier: PrimitiveTier, success: bool) {
        self.record_activation_with_duration(primitive_name, tier, success, None);
    }

    /// Record a primitive activation with duration
    pub fn record_activation_with_duration(
        &mut self,
        primitive_name: &str,
        tier: PrimitiveTier,
        success: bool,
        duration: Option<Duration>,
    ) {
        // Update per-primitive stats
        let stats = self.stats.entry(primitive_name.to_string()).or_default();
        stats.record(success, duration);

        // Update per-tier stats
        let tier_stats = self.tier_stats.entry(tier).or_default();
        tier_stats.activations += 1;
        if success {
            tier_stats.successes += 1;
        }

        // Add to history
        let activation = PrimitiveActivation {
            primitive_name: primitive_name.to_string(),
            tier,
            success,
            duration,
            timestamp: Instant::now(),
            context: self.current_context.clone(),
        };
        self.history.push(activation);

        // Trim history if needed
        if self.history.len() > self.history_size {
            self.history.remove(0);
        }

        self.total_activations += 1;
    }

    /// Record activation using a Primitive reference
    pub fn record_primitive(&mut self, primitive: &Primitive, success: bool) {
        self.record_activation(&primitive.name, primitive.tier, success);
    }

    /// Get statistics for a specific primitive
    pub fn get_stats(&self, primitive_name: &str) -> Option<&PrimitiveStats> {
        self.stats.get(primitive_name)
    }

    /// Get all primitive statistics
    pub fn all_stats(&self) -> &HashMap<String, PrimitiveStats> {
        &self.stats
    }

    /// Get tier statistics
    pub fn get_tier_stats(&self, tier: &PrimitiveTier) -> Option<&TierStats> {
        self.tier_stats.get(tier)
    }

    /// Get top N most used primitives
    pub fn top_primitives(&self, n: usize) -> Vec<(&String, &PrimitiveStats)> {
        let mut sorted: Vec<_> = self.stats.iter().collect();
        sorted.sort_by_key(|x| std::cmp::Reverse(x.1.activations));
        sorted.into_iter().take(n).collect()
    }

    /// Get primitives with low success rate (potential issues)
    pub fn problematic_primitives(&self, threshold: f64) -> Vec<(&String, &PrimitiveStats)> {
        self.stats
            .iter()
            .filter(|(_, s)| s.activations >= 5 && s.success_rate() < threshold)
            .collect()
    }

    /// Get recent activations for a context
    pub fn recent_in_context(&self, context: &str, count: usize) -> Vec<&PrimitiveActivation> {
        self.history
            .iter()
            .rev()
            .filter(|a| a.context == context)
            .take(count)
            .collect()
    }

    /// Get activations per minute
    pub fn activations_per_minute(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64() / 60.0;
        if elapsed > 0.0 {
            self.total_activations as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Primitive Dashboard ===\n\n");

        // Overall stats
        let elapsed = self.start_time.elapsed();
        report.push_str(&format!("Uptime: {:.1}s\n", elapsed.as_secs_f64()));
        report.push_str(&format!("Total Activations: {}\n", self.total_activations));
        report.push_str(&format!(
            "Rate: {:.1}/min\n\n",
            self.activations_per_minute()
        ));

        // Per-tier breakdown
        report.push_str("--- Tier Breakdown ---\n");
        for tier in [
            PrimitiveTier::NSM,
            PrimitiveTier::Mathematical,
            PrimitiveTier::Physical,
            PrimitiveTier::Geometric,
            PrimitiveTier::Strategic,
            PrimitiveTier::MetaCognitive,
            PrimitiveTier::Temporal,
            PrimitiveTier::Compositional,
            PrimitiveTier::Consciousness,
        ] {
            if let Some(stats) = self.tier_stats.get(&tier) {
                let success_rate = if stats.activations > 0 {
                    stats.successes as f64 / stats.activations as f64 * 100.0
                } else {
                    0.0
                };
                report.push_str(&format!(
                    "  {:?}: {} activations ({:.1}% success)\n",
                    tier, stats.activations, success_rate
                ));
            }
        }

        // Top primitives
        report.push_str("\n--- Top 5 Primitives ---\n");
        for (name, stats) in self.top_primitives(5) {
            report.push_str(&format!(
                "  {}: {} uses ({:.1}% success, {:.1}ms avg)\n",
                name,
                stats.activations,
                stats.success_rate() * 100.0,
                stats.avg_duration_ms
            ));
        }

        // Problematic primitives
        let problems = self.problematic_primitives(0.5);
        if !problems.is_empty() {
            report.push_str("\n--- Low Success Rate ---\n");
            for (name, stats) in problems {
                report.push_str(&format!(
                    "  {}: {:.1}% success ({} uses)\n",
                    name,
                    stats.success_rate() * 100.0,
                    stats.activations
                ));
            }
        }

        report
    }

    /// Get a health score (0.0 to 1.0)
    pub fn health_score(&self) -> f64 {
        if self.total_activations == 0 {
            return 1.0; // No data = assume healthy
        }

        // Calculate overall success rate
        let total_successes: usize = self.stats.values().map(|s| s.successes).sum();
        let success_rate = total_successes as f64 / self.total_activations as f64;

        // Penalize if using too few primitives (lack of diversity)
        let diversity = (self.stats.len() as f64 / 20.0).min(1.0); // Expect ~20 primitives

        // Penalize if activations are too slow
        let avg_duration: f64 = self
            .stats
            .values()
            .filter(|s| s.avg_duration_ms > 0.0)
            .map(|s| s.avg_duration_ms)
            .sum::<f64>()
            / self.stats.len().max(1) as f64;
        let speed_score = 1.0 - (avg_duration / 100.0).min(1.0); // 100ms = poor

        // Weighted combination
        success_rate * 0.5 + diversity * 0.3 + speed_score * 0.2
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.stats.clear();
        self.tier_stats.clear();
        self.history.clear();
        self.total_activations = 0;
        self.start_time = Instant::now();
    }
}

impl Default for PrimitiveDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Voice-specific primitive tracker
/// Tracks ATTEND, REMEMBER, INTEND, SALIENCE primitives during voice conversations
#[derive(Debug)]
pub struct VoicePrimitiveTracker {
    dashboard: PrimitiveDashboard,
    /// Currently active attention primitives
    active_attention: Vec<String>,
    /// Current conversation turn
    turn: usize,
}

impl VoicePrimitiveTracker {
    /// Create a new voice primitive tracker
    pub fn new() -> Self {
        let mut dashboard = PrimitiveDashboard::new();
        dashboard.set_context("voice");
        Self {
            dashboard,
            active_attention: Vec::new(),
            turn: 0,
        }
    }

    /// Begin listening - activates ATTEND primitive
    pub fn begin_listening(&mut self) {
        self.dashboard
            .record_activation("ATTEND", PrimitiveTier::Consciousness, true);
        self.active_attention.push("ATTEND".to_string());
    }

    /// Detect speech - activates SALIENCE primitive
    pub fn speech_detected(&mut self, confidence: f32) {
        let success = confidence > 0.5;
        self.dashboard
            .record_activation("SALIENCE", PrimitiveTier::Consciousness, success);
    }

    /// Process transcription - activates BINDING_WINDOW
    pub fn process_transcription(&mut self) {
        self.dashboard
            .record_activation("BINDING_WINDOW", PrimitiveTier::Consciousness, true);
    }

    /// Recall context - activates REMEMBER
    pub fn recall_context(&mut self, success: bool) {
        self.dashboard
            .record_activation("REMEMBER", PrimitiveTier::Consciousness, success);
    }

    /// Generate response - activates INTEND
    pub fn generate_response(&mut self) {
        self.dashboard
            .record_activation("INTEND", PrimitiveTier::Consciousness, true);
        self.dashboard
            .record_activation("DECIDE", PrimitiveTier::Consciousness, true);
    }

    /// Speak response - activates PHENOMENAL_BINDING
    pub fn begin_speaking(&mut self) {
        self.dashboard
            .record_activation("PHENOMENAL_BINDING", PrimitiveTier::Consciousness, true);
    }

    /// End turn - cleanup
    pub fn end_turn(&mut self) {
        self.active_attention.clear();
        self.turn += 1;
    }

    /// Get conversation statistics
    pub fn stats(&self) -> &PrimitiveDashboard {
        &self.dashboard
    }

    /// Get current turn number
    pub fn turn_number(&self) -> usize {
        self.turn
    }

    /// Summary for voice session
    pub fn voice_summary(&self) -> String {
        let mut summary = format!("=== Voice Session ({} turns) ===\n", self.turn);

        // Get voice-specific primitives
        let voice_primitives = [
            "ATTEND",
            "SALIENCE",
            "REMEMBER",
            "INTEND",
            "DECIDE",
            "BINDING_WINDOW",
            "PHENOMENAL_BINDING",
        ];

        for prim in &voice_primitives {
            if let Some(stats) = self.dashboard.get_stats(prim) {
                summary.push_str(&format!(
                    "  {}: {} activations ({:.0}% success)\n",
                    prim,
                    stats.activations,
                    stats.success_rate() * 100.0
                ));
            }
        }

        summary
    }
}

impl Default for VoicePrimitiveTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_dashboard_basic() {
        let mut dashboard = PrimitiveDashboard::new();

        // Record some activations
        dashboard.record_activation("PART_OF", PrimitiveTier::Compositional, true);
        dashboard.record_activation("PART_OF", PrimitiveTier::Compositional, true);
        dashboard.record_activation("CAUSE", PrimitiveTier::Physical, true);
        dashboard.record_activation("CAUSE", PrimitiveTier::Physical, false);

        assert_eq!(dashboard.total_activations, 4);

        let part_of_stats = dashboard.get_stats("PART_OF").unwrap();
        assert_eq!(part_of_stats.activations, 2);
        assert_eq!(part_of_stats.success_rate(), 1.0);

        let cause_stats = dashboard.get_stats("CAUSE").unwrap();
        assert_eq!(cause_stats.activations, 2);
        assert_eq!(cause_stats.success_rate(), 0.5);
    }

    #[test]
    fn test_top_primitives() {
        let mut dashboard = PrimitiveDashboard::new();

        // Record with different frequencies
        for _ in 0..10 {
            dashboard.record_activation("ATTEND", PrimitiveTier::Consciousness, true);
        }
        for _ in 0..5 {
            dashboard.record_activation("REMEMBER", PrimitiveTier::Consciousness, true);
        }
        for _ in 0..3 {
            dashboard.record_activation("INTEND", PrimitiveTier::Consciousness, true);
        }

        let top = dashboard.top_primitives(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "ATTEND");
        assert_eq!(top[1].0, "REMEMBER");
    }

    #[test]
    fn test_problematic_primitives() {
        let mut dashboard = PrimitiveDashboard::new();

        // Good primitive
        for _ in 0..10 {
            dashboard.record_activation("GOOD", PrimitiveTier::NSM, true);
        }

        // Bad primitive (low success rate)
        for i in 0..10 {
            dashboard.record_activation("BAD", PrimitiveTier::NSM, i < 3); // 30% success
        }

        let problems = dashboard.problematic_primitives(0.5);
        assert_eq!(problems.len(), 1);
        assert_eq!(problems[0].0, "BAD");
    }

    #[test]
    fn test_voice_primitive_tracker() {
        let mut tracker = VoicePrimitiveTracker::new();

        // Simulate a voice turn
        tracker.begin_listening();
        tracker.speech_detected(0.9);
        tracker.process_transcription();
        tracker.recall_context(true);
        tracker.generate_response();
        tracker.begin_speaking();
        tracker.end_turn();

        assert_eq!(tracker.turn_number(), 1);

        let stats = tracker.stats();
        assert!(stats.get_stats("ATTEND").is_some());
        assert!(stats.get_stats("SALIENCE").is_some());
        assert!(stats.get_stats("INTEND").is_some());
    }

    #[test]
    fn test_dashboard_summary() {
        let mut dashboard = PrimitiveDashboard::new();

        dashboard.record_activation("ATTEND", PrimitiveTier::Consciousness, true);
        dashboard.record_activation("REMEMBER", PrimitiveTier::Consciousness, true);

        let summary = dashboard.summary();
        assert!(summary.contains("Primitive Dashboard"));
        assert!(summary.contains("Total Activations: 2"));
    }

    #[test]
    fn test_health_score() {
        let mut dashboard = PrimitiveDashboard::new();

        // All successes should give high health
        for _ in 0..20 {
            dashboard.record_activation("ATTEND", PrimitiveTier::Consciousness, true);
        }

        let health = dashboard.health_score();
        assert!(health > 0.5); // Should be healthy
    }
}
