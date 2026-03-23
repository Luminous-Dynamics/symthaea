// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flow state detection and management for the cognitive loop.
//!
//! Detects optimal cognitive engagement (flow state) based on sustained focus,
//! low prediction error, and high temporal coherence. Flow state boosts learning
//! efficiency and signals peak cognitive performance.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::dynamics::temporal_signatures::ConsciousnessPattern;

/// Flow state - optimal cognitive engagement
///
/// Detected when there is sustained focus, low prediction error,
/// and high temporal coherence. Flow state boosts learning efficiency
/// and signals peak cognitive performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowState {
    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow intensity (0.0 to 1.0)
    /// Higher = deeper flow state
    pub intensity: f32,

    /// Consecutive cycles in flow-compatible state
    pub streak: u32,

    /// Average prediction error during flow detection window
    pub avg_error: f32,

    /// Average coherence during flow detection window
    pub avg_coherence: f32,

    /// Learning rate boost when in flow (1.0 to 2.0)
    pub learning_boost: f32,

    /// Attention enhancement when in flow
    pub attention_boost: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING - Time Context for Flow States
    // ═══════════════════════════════════════════════════════════════════════════
    /// Timestamp when flow state started (if in flow)
    /// Note: Not serialized as Instant is monotonic/non-portable
    #[serde(skip)]
    pub flow_started_at: Option<Instant>,

    /// Total time spent in flow during this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average duration of flow periods (seconds)
    pub avg_flow_duration_secs: f32,
}

impl Default for FlowState {
    fn default() -> Self {
        Self {
            in_flow: false,
            intensity: 0.0,
            streak: 0,
            avg_error: 0.5,
            avg_coherence: 0.5,
            learning_boost: 1.0,
            attention_boost: 1.0,
            // Temporal encoding defaults
            flow_started_at: None,
            total_flow_time_secs: 0.0,
            flow_periods: 0,
            avg_flow_duration_secs: 0.0,
        }
    }
}

impl FlowState {
    /// Minimum streak for flow state entry
    pub const FLOW_ENTRY_STREAK: u32 = 5;
    /// Error threshold for flow eligibility
    const FLOW_ERROR_THRESHOLD: f32 = 0.25;
    /// Coherence threshold for flow eligibility
    const FLOW_COHERENCE_THRESHOLD: f32 = 0.6;
    /// Confidence threshold for flow eligibility
    const FLOW_CONFIDENCE_THRESHOLD: f32 = 0.5;

    /// Update flow state based on current metrics
    pub fn update(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        coherence: f32,
        prediction_confidence: f32,
    ) {
        // Check if current state is flow-compatible
        let is_flow_compatible = matches!(
            pattern,
            ConsciousnessPattern::Focused | ConsciousnessPattern::Contemplative
        ) && prediction_error < Self::FLOW_ERROR_THRESHOLD
            && coherence > Self::FLOW_COHERENCE_THRESHOLD
            && prediction_confidence > Self::FLOW_CONFIDENCE_THRESHOLD;

        // Update running averages (EMA)
        let alpha = super::thresholds::EMA_ALPHA_FLOW;
        self.avg_error = self.avg_error * (1.0 - alpha) + prediction_error * alpha;
        self.avg_coherence = self.avg_coherence * (1.0 - alpha) + coherence * alpha;

        if is_flow_compatible {
            self.streak += 1;

            // Enter flow state after sustained focus
            if self.streak >= Self::FLOW_ENTRY_STREAK {
                self.in_flow = true;

                // Intensity grows with streak (caps at 1.0)
                // Use saturating_sub to prevent underflow, then safe cast via f64
                self.intensity = (self.streak.saturating_sub(Self::FLOW_ENTRY_STREAK) as f64 / 10.0)
                    .min(1.0) as f32;

                // Boost learning when in flow (up to 50% boost at max intensity)
                self.learning_boost = 1.0 + 0.5 * self.intensity;

                // Enhance attention (up to 30% boost)
                self.attention_boost = 1.0 + 0.3 * self.intensity;
            }
        } else {
            // Exit flow or reduce streak
            if self.in_flow {
                // Grace period: don't exit immediately
                if self.streak > 0 {
                    self.streak = self.streak.saturating_sub(2);
                }
                if self.streak < Self::FLOW_ENTRY_STREAK / 2 {
                    self.in_flow = false;
                    self.intensity = 0.0;
                    self.learning_boost = 1.0;
                    self.attention_boost = 1.0;
                }
            } else {
                self.streak = 0;
            }
        }
    }

    /// Update flow state with adaptive thresholds from self-reflection
    ///
    /// This allows the meta-learning system to adjust flow entry criteria.
    pub fn update_with_thresholds(
        &mut self,
        pattern: ConsciousnessPattern,
        prediction_error: f32,
        coherence: f32,
        prediction_confidence: f32,
        error_threshold: f32,
        coherence_threshold: f32,
    ) {
        // Check if current state is flow-compatible using adaptive thresholds
        let is_flow_compatible = matches!(
            pattern,
            ConsciousnessPattern::Focused | ConsciousnessPattern::Contemplative
        ) && prediction_error < error_threshold
            && coherence > coherence_threshold
            && prediction_confidence > Self::FLOW_CONFIDENCE_THRESHOLD;

        // Update running averages (EMA)
        let alpha = super::thresholds::EMA_ALPHA_FLOW;
        self.avg_error = self.avg_error * (1.0 - alpha) + prediction_error * alpha;
        self.avg_coherence = self.avg_coherence * (1.0 - alpha) + coherence * alpha;

        if is_flow_compatible {
            self.streak += 1;

            // Enter flow state after sustained focus
            if self.streak >= Self::FLOW_ENTRY_STREAK {
                // Track temporal: entering flow
                let was_in_flow = self.in_flow;
                self.in_flow = true;

                // Start flow timer if just entering
                if !was_in_flow {
                    self.flow_started_at = Some(Instant::now());
                    self.flow_periods = self.flow_periods.saturating_add(1);
                }

                // Intensity grows with streak (caps at 1.0)
                // Use saturating_sub to prevent underflow, then safe cast via f64
                self.intensity = (self.streak.saturating_sub(Self::FLOW_ENTRY_STREAK) as f64 / 10.0)
                    .min(1.0) as f32;

                // Boost learning when in flow (up to 50% boost at max intensity)
                self.learning_boost = 1.0 + 0.5 * self.intensity;

                // Enhance attention (up to 30% boost)
                self.attention_boost = 1.0 + 0.3 * self.intensity;
            }
        } else {
            // Exit flow or reduce streak
            if self.in_flow {
                // Grace period: don't exit immediately
                if self.streak > 0 {
                    self.streak = self.streak.saturating_sub(2);
                }
                if self.streak < Self::FLOW_ENTRY_STREAK / 2 {
                    // Track temporal: exiting flow
                    if let Some(started) = self.flow_started_at.take() {
                        let duration = started.elapsed().as_secs_f32();
                        self.total_flow_time_secs += duration;

                        // Update average duration (safe division with max(1))
                        if self.flow_periods > 0 {
                            self.avg_flow_duration_secs =
                                self.total_flow_time_secs / self.flow_periods.max(1) as f32;
                        }
                    }

                    self.in_flow = false;
                    self.intensity = 0.0;
                    self.learning_boost = 1.0;
                    self.attention_boost = 1.0;
                }
            } else {
                self.streak = 0;
            }
        }
    }

    /// Reset flow state
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get effective learning rate multiplier including flow boost
    pub fn effective_learning_multiplier(&self, base_multiplier: f32) -> f32 {
        base_multiplier * self.learning_boost
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING METHODS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Get current flow duration in seconds (if in flow)
    pub fn current_flow_duration_secs(&self) -> Option<f32> {
        self.flow_started_at
            .map(|started| started.elapsed().as_secs_f32())
    }

    /// Get total time spent in flow (including current session)
    pub fn total_flow_time_with_current(&self) -> f32 {
        let current = self.current_flow_duration_secs().unwrap_or(0.0);
        self.total_flow_time_secs + current
    }

    /// Get the timestamp when current flow started
    pub fn flow_started(&self) -> Option<Instant> {
        self.flow_started_at
    }

    /// Get flow statistics summary
    pub fn temporal_summary(&self) -> FlowTemporalSummary {
        FlowTemporalSummary {
            total_flow_time_secs: self.total_flow_time_with_current(),
            flow_periods: self.flow_periods,
            avg_flow_duration_secs: self.avg_flow_duration_secs,
            current_flow_duration_secs: self.current_flow_duration_secs(),
            is_in_flow: self.in_flow,
        }
    }
}

/// Summary of flow state temporal statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowTemporalSummary {
    /// Total time spent in flow during this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average duration of flow periods (seconds)
    pub avg_flow_duration_secs: f32,

    /// Current flow duration if in flow (seconds)
    pub current_flow_duration_secs: Option<f32>,

    /// Whether currently in flow
    pub is_in_flow: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLOSED LEARNING LOOP - Strategy-Based Behavioral Adaptation
// ═══════════════════════════════════════════════════════════════════════════════

/// Response strategy selected by the closed learning loop
///
/// Based on CLOSED_LEARNING_LOOP.md - strategies are selected based on:
/// 1. Q-learning from past interactions
/// 2. Previous reward (stick with success, avoid failure)
/// 3. Φ-gating (high Φ → Exploratory, low Φ → Supportive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ResponseStrategy {
    /// Elaborate explanations with detail
    Detailed,
    /// Brief, direct answers
    Concise,
    /// Ask clarifying questions
    Clarifying,
    /// Acknowledge and validate
    #[default]
    Supportive,
    /// Offer new perspectives
    Exploratory,
}

impl std::fmt::Display for ResponseStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ResponseStrategy {
    /// Return the strategy name as a static string, matching Debug output.
    /// Avoids `format!("{:?}", strategy)` allocation on the hot path.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Detailed => "Detailed",
            Self::Concise => "Concise",
            Self::Clarifying => "Clarifying",
            Self::Supportive => "Supportive",
            Self::Exploratory => "Exploratory",
        }
    }

    /// Get the opposite strategy (for switching after negative feedback)
    pub fn opposite(self) -> Self {
        match self {
            Self::Detailed => Self::Concise,
            Self::Concise => Self::Detailed,
            Self::Clarifying => Self::Supportive,
            Self::Supportive => Self::Exploratory,
            Self::Exploratory => Self::Clarifying,
        }
    }

    /// Get description of strategy
    pub fn description(&self) -> &'static str {
        match self {
            Self::Detailed => "Elaborate explanations with full context",
            Self::Concise => "Brief, direct responses",
            Self::Clarifying => "Ask questions to understand better",
            Self::Supportive => "Acknowledge and validate",
            Self::Exploratory => "Offer novel perspectives and connections",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState default and construction
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_state_default_values() {
        let fs = FlowState::default();
        assert!(!fs.in_flow);
        assert_eq!(fs.intensity, 0.0);
        assert_eq!(fs.streak, 0);
        assert_eq!(fs.avg_error, 0.5);
        assert_eq!(fs.avg_coherence, 0.5);
        assert_eq!(fs.learning_boost, 1.0);
        assert_eq!(fs.attention_boost, 1.0);
        assert!(fs.flow_started_at.is_none());
        assert_eq!(fs.total_flow_time_secs, 0.0);
        assert_eq!(fs.flow_periods, 0);
        assert_eq!(fs.avg_flow_duration_secs, 0.0);
    }

    #[test]
    fn flow_state_constants() {
        assert_eq!(FlowState::FLOW_ENTRY_STREAK, 5);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::update — streak accumulation and flow entry
    // ═══════════════════════════════════════════════════════════════════════

    /// Helper: apply N flow-compatible updates using `update()`
    fn pump_flow_compatible(fs: &mut FlowState, n: usize) {
        for _ in 0..n {
            fs.update(ConsciousnessPattern::Focused, 0.1, 0.8, 0.7);
        }
    }

    #[test]
    fn flow_streak_increments_on_compatible_input() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, 3);
        assert_eq!(fs.streak, 3);
        assert!(
            !fs.in_flow,
            "should not enter flow before FLOW_ENTRY_STREAK"
        );
    }

    #[test]
    fn flow_enters_at_entry_streak() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, FlowState::FLOW_ENTRY_STREAK as usize);
        assert!(fs.in_flow);
        // At exactly FLOW_ENTRY_STREAK, intensity = (5-5)/10 = 0.0
        assert_eq!(fs.intensity, 0.0);
    }

    #[test]
    fn flow_intensity_grows_beyond_entry_streak() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, (FlowState::FLOW_ENTRY_STREAK + 5) as usize);
        assert!(fs.in_flow);
        // intensity = (10-5)/10 = 0.5
        assert!((fs.intensity - 0.5).abs() < 1e-5);
        assert!((fs.learning_boost - 1.25).abs() < 1e-5); // 1.0 + 0.5 * 0.5
        assert!((fs.attention_boost - 1.15).abs() < 1e-5); // 1.0 + 0.3 * 0.5
    }

    #[test]
    fn flow_intensity_caps_at_one() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, (FlowState::FLOW_ENTRY_STREAK + 20) as usize);
        assert!(fs.in_flow);
        assert!((fs.intensity - 1.0).abs() < 1e-5);
        assert!((fs.learning_boost - 1.5).abs() < 1e-5); // max boost
        assert!((fs.attention_boost - 1.3).abs() < 1e-5);
    }

    #[test]
    fn flow_incompatible_pattern_resets_streak_when_not_in_flow() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, 3);
        assert_eq!(fs.streak, 3);
        // Incompatible pattern (Exploratory)
        fs.update(ConsciousnessPattern::Exploratory, 0.1, 0.8, 0.7);
        assert_eq!(fs.streak, 0);
    }

    #[test]
    fn flow_grace_period_when_in_flow() {
        let mut fs = FlowState::default();
        // Enter flow
        pump_flow_compatible(&mut fs, (FlowState::FLOW_ENTRY_STREAK + 2) as usize);
        assert!(fs.in_flow);
        assert_eq!(fs.streak, 7);
        // One bad cycle: streak decreases by 2 but still >= FLOW_ENTRY_STREAK/2
        fs.update(ConsciousnessPattern::Exploratory, 0.5, 0.3, 0.2);
        assert_eq!(fs.streak, 5); // 7 - 2
        assert!(fs.in_flow, "should remain in flow during grace period");
    }

    #[test]
    fn flow_exits_when_streak_drops_below_half_entry() {
        let mut fs = FlowState::default();
        // Enter flow with minimal streak
        pump_flow_compatible(&mut fs, FlowState::FLOW_ENTRY_STREAK as usize);
        assert!(fs.in_flow);
        assert_eq!(fs.streak, 5);
        // Bad cycle: streak drops from 5 to 3, which is >= 5/2=2, stays in flow
        fs.update(ConsciousnessPattern::Resting, 0.5, 0.3, 0.2);
        assert_eq!(fs.streak, 3);
        assert!(fs.in_flow);
        // Another bad cycle: streak drops from 3 to 1, which is < 5/2=2, exits flow
        fs.update(ConsciousnessPattern::Resting, 0.5, 0.3, 0.2);
        assert_eq!(fs.streak, 1);
        assert!(!fs.in_flow);
        assert_eq!(fs.intensity, 0.0);
        assert_eq!(fs.learning_boost, 1.0);
        assert_eq!(fs.attention_boost, 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::update — EMA averages
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_ema_updates_error_and_coherence() {
        let mut fs = FlowState::default();
        // Default avg_error=0.5, avg_coherence=0.5
        // After one update with error=0.1, coherence=0.9:
        // avg_error = 0.5*0.8 + 0.1*0.2 = 0.42
        // avg_coherence = 0.5*0.8 + 0.9*0.2 = 0.58
        fs.update(ConsciousnessPattern::Focused, 0.1, 0.9, 0.7);
        assert!((fs.avg_error - 0.42).abs() < 1e-5);
        assert!((fs.avg_coherence - 0.58).abs() < 1e-5);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::update — flow-incompatible reasons
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_high_error_prevents_flow() {
        let mut fs = FlowState::default();
        for _ in 0..10 {
            // Focused + good coherence + good confidence, but error too high
            fs.update(ConsciousnessPattern::Focused, 0.3, 0.8, 0.7);
        }
        assert!(!fs.in_flow);
        assert_eq!(fs.streak, 0);
    }

    #[test]
    fn flow_low_coherence_prevents_flow() {
        let mut fs = FlowState::default();
        for _ in 0..10 {
            // Focused + low error + good confidence, but coherence too low
            fs.update(ConsciousnessPattern::Focused, 0.1, 0.4, 0.7);
        }
        assert!(!fs.in_flow);
    }

    #[test]
    fn flow_low_confidence_prevents_flow() {
        let mut fs = FlowState::default();
        for _ in 0..10 {
            // Focused + low error + good coherence, but confidence too low
            fs.update(ConsciousnessPattern::Focused, 0.1, 0.8, 0.3);
        }
        assert!(!fs.in_flow);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::update_with_thresholds — adaptive thresholds + temporal
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_with_thresholds_respects_custom_error_threshold() {
        let mut fs = FlowState::default();
        // With default threshold (0.25), error=0.3 would fail.
        // With relaxed threshold (0.5), it should succeed.
        for _ in 0..6 {
            fs.update_with_thresholds(
                ConsciousnessPattern::Focused,
                0.3, // error
                0.8, // coherence
                0.7, // confidence
                0.5, // relaxed error threshold
                0.6, // coherence threshold
            );
        }
        assert!(fs.in_flow);
    }

    #[test]
    fn flow_with_thresholds_tracks_flow_periods() {
        let mut fs = FlowState::default();
        // Enter flow
        for _ in 0..6 {
            fs.update_with_thresholds(
                ConsciousnessPattern::Contemplative,
                0.1,
                0.8,
                0.7,
                0.25,
                0.6,
            );
        }
        assert!(fs.in_flow);
        assert_eq!(fs.flow_periods, 1);
        assert!(fs.flow_started_at.is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::reset
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_reset_returns_to_default() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, 10);
        assert!(fs.in_flow);
        fs.reset();
        assert!(!fs.in_flow);
        assert_eq!(fs.streak, 0);
        assert_eq!(fs.intensity, 0.0);
        assert_eq!(fs.learning_boost, 1.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState::effective_learning_multiplier
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn effective_learning_multiplier_no_flow() {
        let fs = FlowState::default();
        assert!((fs.effective_learning_multiplier(0.01) - 0.01).abs() < 1e-7);
    }

    #[test]
    fn effective_learning_multiplier_in_flow() {
        let mut fs = FlowState::default();
        pump_flow_compatible(&mut fs, (FlowState::FLOW_ENTRY_STREAK + 10) as usize);
        // intensity = min((15-5)/10, 1.0) = 1.0
        // learning_boost = 1.5
        assert!((fs.effective_learning_multiplier(0.01) - 0.015).abs() < 1e-7);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlowState — temporal encoding methods
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn current_flow_duration_none_when_not_in_flow() {
        let fs = FlowState::default();
        assert!(fs.current_flow_duration_secs().is_none());
    }

    #[test]
    fn total_flow_time_with_current_no_flow() {
        let fs = FlowState::default();
        assert_eq!(fs.total_flow_time_with_current(), 0.0);
    }

    #[test]
    fn flow_started_none_by_default() {
        let fs = FlowState::default();
        assert!(fs.flow_started().is_none());
    }

    #[test]
    fn temporal_summary_default() {
        let fs = FlowState::default();
        let summary = fs.temporal_summary();
        assert!(!summary.is_in_flow);
        assert_eq!(summary.flow_periods, 0);
        assert_eq!(summary.avg_flow_duration_secs, 0.0);
        assert!(summary.current_flow_duration_secs.is_none());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ResponseStrategy
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn response_strategy_default_is_supportive() {
        assert_eq!(ResponseStrategy::default(), ResponseStrategy::Supportive);
    }

    #[test]
    fn response_strategy_as_str() {
        assert_eq!(ResponseStrategy::Detailed.as_str(), "Detailed");
        assert_eq!(ResponseStrategy::Concise.as_str(), "Concise");
        assert_eq!(ResponseStrategy::Clarifying.as_str(), "Clarifying");
        assert_eq!(ResponseStrategy::Supportive.as_str(), "Supportive");
        assert_eq!(ResponseStrategy::Exploratory.as_str(), "Exploratory");
    }

    #[test]
    fn response_strategy_opposite_is_symmetric_pair() {
        // Detailed <-> Concise
        assert_eq!(
            ResponseStrategy::Detailed.opposite(),
            ResponseStrategy::Concise
        );
        assert_eq!(
            ResponseStrategy::Concise.opposite(),
            ResponseStrategy::Detailed
        );
        // Clarifying -> Supportive -> Exploratory -> Clarifying (cycle)
        assert_eq!(
            ResponseStrategy::Clarifying.opposite(),
            ResponseStrategy::Supportive
        );
        assert_eq!(
            ResponseStrategy::Supportive.opposite(),
            ResponseStrategy::Exploratory
        );
        assert_eq!(
            ResponseStrategy::Exploratory.opposite(),
            ResponseStrategy::Clarifying
        );
    }

    #[test]
    fn response_strategy_double_opposite() {
        // Detailed -> Concise -> Detailed
        assert_eq!(
            ResponseStrategy::Detailed.opposite().opposite(),
            ResponseStrategy::Detailed
        );
        // The Clarifying->Supportive->Exploratory cycle has period 3
        assert_eq!(
            ResponseStrategy::Clarifying
                .opposite()
                .opposite()
                .opposite(),
            ResponseStrategy::Clarifying
        );
    }

    #[test]
    fn response_strategy_description_non_empty() {
        let strategies = [
            ResponseStrategy::Detailed,
            ResponseStrategy::Concise,
            ResponseStrategy::Clarifying,
            ResponseStrategy::Supportive,
            ResponseStrategy::Exploratory,
        ];
        for s in &strategies {
            assert!(!s.description().is_empty(), "{:?} has empty description", s);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Serialization round-trip
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn flow_state_serialization_roundtrip() {
        let fs = FlowState::default();
        let json = serde_json::to_string(&fs).expect("serialize");
        let deserialized: FlowState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.in_flow, fs.in_flow);
        assert_eq!(deserialized.intensity, fs.intensity);
        assert_eq!(deserialized.streak, fs.streak);
    }

    #[test]
    fn response_strategy_serialization_roundtrip() {
        let strategy = ResponseStrategy::Exploratory;
        let json = serde_json::to_string(&strategy).expect("serialize");
        let deserialized: ResponseStrategy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, strategy);
    }
}
