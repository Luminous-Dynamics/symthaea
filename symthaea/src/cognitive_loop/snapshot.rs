//! Consciousness snapshot - unified cognitive metrics dashboard
//!
//! Provides a single point of observation for the entire cognitive state,
//! making it easy to monitor, log, or expose via API.

use serde::{Deserialize, Serialize};

use super::drives::SelfAssessment;
use super::routing::CognitiveDepth;
use super::ActionHint;
use crate::consciousness::consciousness_unification::{EmotionalPattern, UnifiedEmotion};
use crate::dynamics::temporal_signatures::ConsciousnessPattern;

// ============================================================================
// CONSCIOUSNESS SNAPSHOT - Unified Dashboard
// ============================================================================

/// Unified consciousness snapshot - aggregates all cognitive metrics
///
/// This provides a single point of observation for the entire cognitive state,
/// making it easy to monitor, log, or expose via API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSnapshot {
    // ===== Core Metrics =====
    /// Timestamp of this snapshot (cycle count)
    pub cycle: usize,

    /// Overall consciousness level (0.0 to 1.0)
    /// Computed from prediction confidence, coherence, and flow
    pub consciousness_level: f32,

    /// Current consciousness pattern
    pub pattern: ConsciousnessPattern,

    /// Pattern classification confidence
    pub pattern_confidence: f32,

    // ===== Prediction & Learning =====
    /// Current prediction error
    pub prediction_error: f32,

    /// Prediction confidence (decays during uncertainty)
    pub prediction_confidence: f32,

    /// Whether predictions should be trusted
    pub predictions_trustworthy: bool,

    /// Effective learning rate (after all modulations)
    pub effective_learning_rate: f32,

    /// Learning effectiveness score from self-reflection
    pub learning_effectiveness: f32,

    // ===== Flow State =====
    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow intensity (0.0 to 1.0)
    pub flow_intensity: f32,

    /// Consecutive flow-compatible cycles
    pub flow_streak: u32,

    /// Learning boost from flow
    pub flow_learning_boost: f32,

    // ===== Curiosity & Exploration =====
    /// Boredom level (0.0 to 1.0)
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    pub exploration_urge: f32,

    /// Whether curiosity is triggering exploration
    pub exploring: bool,

    /// Novelty bonus for learning
    pub novelty_bonus: f32,

    // ===== Emotional State =====
    /// Emotional valence (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Emotional arousal (0.0 to 1.0)
    pub emotional_arousal: f32,

    /// Whether input has significant emotional content
    pub has_emotional_content: bool,

    /// Emotion-suggested pattern nudge
    pub emotion_nudge: Option<ConsciousnessPattern>,

    // ===== Self-Reflection =====
    /// Self-assessment from meta-learning
    pub self_assessment: SelfAssessment,

    /// Number of reflection cycles performed
    pub reflection_count: u64,

    /// Threshold adjustments made
    pub adjustments_made: u32,

    /// Cycles until next reflection
    pub next_reflection_in: u32,

    // ===== Adaptive Behavior =====
    /// Recommended action
    pub action_hint: ActionHint,

    /// Speech rate multiplier
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier
    pub pause_multiplier: f32,

    /// Whether learning is paused
    pub learning_paused: bool,

    // ===== Adapted Thresholds =====
    /// Adapted flow error threshold
    pub flow_threshold: f32,

    /// Adapted boredom threshold
    pub boredom_threshold: f32,

    /// Adapted trust threshold
    pub trust_threshold: f32,

    // ===== Temporal Coherence =====
    /// Temporal coherence from CfC
    pub temporal_coherence: f32,

    /// Tau trajectory mean
    pub tau_mean: f32,

    /// Tau trajectory trend
    pub tau_trend: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE FIELDS
    // ═══════════════════════════════════════════════════════════════════════════
    /// Current cognitive depth (Reflex/Cortical/DeepThought)
    pub cognitive_depth: CognitiveDepth,

    /// Unified Φ from ConsciousnessUnificationEngine
    pub unified_phi: f32,

    /// Unified emotional valence (VAD-based, -1.0 to 1.0)
    pub unified_valence: f32,

    /// Unified emotional arousal (VAD-based, 0.0 to 1.0)
    pub unified_arousal: f32,

    /// Unified emotional dominance (VAD-based, -1.0 to 1.0)
    pub unified_dominance: f32,

    /// Discrete emotion from unified EmotionalBridge
    pub unified_discrete_emotion: Option<UnifiedEmotion>,

    /// Emotional pattern (Stable/Escalating/Calming/Volatile)
    pub emotional_pattern: EmotionalPattern,

    /// Emotional description in natural language
    pub emotional_description: String,

    // ═══════════════════════════════════════════════════════════════════════════
    // TEMPORAL ENCODING FIELDS
    // ═══════════════════════════════════════════════════════════════════════════
    /// Snapshot creation timestamp (monotonic, for relative time)
    pub snapshot_timestamp_nanos: u64,

    /// Current flow duration if in flow (seconds)
    pub current_flow_duration_secs: Option<f32>,

    /// Total time spent in flow this session (seconds)
    pub total_flow_time_secs: f32,

    /// Number of distinct flow periods
    pub flow_periods: u32,

    /// Average flow period duration (seconds)
    pub avg_flow_duration_secs: f32,

    // ===== FEP Active Inference =====
    /// FEP variational free energy
    pub fep_free_energy: f64,

    /// FEP precision estimate
    pub fep_precision: f64,
}

impl ConsciousnessSnapshot {
    /// Compute overall consciousness level from components
    pub(super) fn compute_consciousness_level(
        prediction_confidence: f32,
        temporal_coherence: f32,
        flow_intensity: f32,
        pattern_confidence: f32,
    ) -> f32 {
        // Weighted combination of key indicators
        let confidence_contrib = prediction_confidence * 0.3;
        let coherence_contrib = temporal_coherence * 0.25;
        let flow_contrib = flow_intensity * 0.2;
        let pattern_contrib = pattern_confidence * 0.25;

        (confidence_contrib + coherence_contrib + flow_contrib + pattern_contrib).clamp(0.0, 1.0)
    }

    /// Get a concise status string
    pub fn status(&self) -> String {
        let flow_status = if self.in_flow { "FLOW" } else { "---" };
        let explore_status = if self.exploring { "EXPLORE" } else { "---" };

        format!(
            "[L:{:.2}] {:?} | {} {} | Conf:{:.2} Err:{:.2}",
            self.consciousness_level,
            self.pattern,
            flow_status,
            explore_status,
            self.prediction_confidence,
            self.prediction_error,
        )
    }

    /// Check if system is in an optimal state
    pub fn is_optimal(&self) -> bool {
        self.self_assessment == SelfAssessment::Optimal
            || (self.in_flow && self.prediction_confidence > 0.6)
    }

    /// Check if system needs attention (struggling or stagnating)
    pub fn needs_attention(&self) -> bool {
        matches!(
            self.self_assessment,
            SelfAssessment::Struggling
                | SelfAssessment::Stagnating
                | SelfAssessment::NeedsCalibration
        )
    }

    /// Get the dominant concern (what needs most attention)
    pub fn dominant_concern(&self) -> Option<&'static str> {
        if self.self_assessment == SelfAssessment::Struggling {
            Some("High prediction error - system is struggling")
        } else if self.self_assessment == SelfAssessment::Stagnating {
            Some("Low error but no exploration - system is stagnating")
        } else if self.boredom > 0.7 && !self.exploring {
            Some("High boredom - needs novel input")
        } else if self.prediction_confidence < 0.3 {
            Some("Low confidence - predictions unreliable")
        } else if self.self_assessment == SelfAssessment::NeedsCalibration {
            Some("Many adjustments made - consider manual review")
        } else {
            None
        }
    }

    /// Get recommended actions based on current state
    pub fn recommended_actions(&self) -> Vec<&'static str> {
        let mut actions = Vec::new();

        match self.action_hint {
            ActionHint::SlowDown => actions.push("Reduce input rate"),
            ActionHint::SpeedUp => actions.push("Can increase input rate"),
            ActionHint::Stabilize => actions.push("Maintain current input"),
            ActionHint::Explore => actions.push("Introduce novel inputs"),
            ActionHint::SeekInput => actions.push("System needs more input"),
            ActionHint::Continue => {}
        }

        if self.boredom > 0.5 && !self.exploring {
            actions.push("Consider varying input content");
        }

        if !self.predictions_trustworthy {
            actions.push("Predictions currently unreliable");
        }

        if self.in_flow {
            actions.push("In flow state - optimal for learning");
        }

        actions
    }
}
