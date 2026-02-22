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
        let alpha = 0.2;
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
        let alpha = 0.2;
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
                    self.flow_periods += 1;
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
pub(crate) struct FlowTemporalSummary {
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
