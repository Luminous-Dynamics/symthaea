// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness snapshot - unified cognitive metrics dashboard
//!
//! Provides a single point of observation for the entire cognitive state,
//! making it easy to monitor, log, or expose via API.

use serde::{Deserialize, Serialize};

use super::ActionHint;
use super::drives::SelfAssessment;
use super::routing::CognitiveDepth;
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
    pub unified_psi: f32,

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

    // ═══════════════════════════════════════════════════════════════════════════
    // SPECTRAL MIP + CONSCIOUSNESS ENRICHMENT
    // ═══════════════════════════════════════════════════════════════════════════
    /// Spectral MIP Phi — O(n³) Minimum Information Partition (None = not yet computed).
    pub spectral_mip_phi: Option<f64>,

    /// Whether `SpectralMIPFinder` has adapted its tracked dimension selection at
    /// least once (`adapt()` runs every 94 cycles in production, per `measure.rs`).
    /// Added 2026-07-05: `is_adapted()`/`active_dim_indices()` had zero non-test call
    /// sites anywhere — the adaptation ran silently in production with no telemetry
    /// visibility since the mechanism was added. This closes that gap; it does not
    /// change adaptation behavior itself.
    pub spectral_mip_adapted: bool,

    /// How many of the full HDC dimension space are currently tracked by the adapted
    /// selection, if adaptation has happened yet (`None` before the first `adapt()`).
    /// A count, not the full index list — this is a snapshot for telemetry/dashboards,
    /// not a debugging dump; use `SpectralMIPFinder::active_dim_indices()` directly if
    /// the exact indices are ever needed.
    pub spectral_mip_active_dim_count: Option<usize>,

    /// Harmonies alignment score (0.0–1.0) from Eight Harmonies integrator.
    pub harmonies_alignment: f32,

    /// Empathic compassion level (0.0–1.0) from empathic unification.
    pub empathic_compassion: f64,

    /// Sigma (backward-compat alias for spectral MIP phi, used by memory coordinator).
    pub sigma: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════════════
    // INTEGRITY
    // ═══════════════════════════════════════════════════════════════════════════
    /// Whether a critical integrity anomaly was detected (attestation failure,
    /// canary corruption). When true, consciousness metrics may be untrustworthy.
    /// Feature-gated behind `integrity`; defaults to false.
    #[serde(default)]
    pub integrity_critical: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // MESH NETWORK
    // ═══════════════════════════════════════════════════════════════════════════
    /// Whether the network is in a critical state (all radio tiers down or sustained jamming).
    /// Feature-gated behind `mesh`; defaults to false.
    #[serde(default)]
    pub network_critical: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // PERFORMANCE METRICS
    // ═══════════════════════════════════════════════════════════════════════════
    /// Average cycle time in microseconds (EMA).
    pub avg_cycle_time_us: f32,

    /// Current cycles per second (Hz).
    pub cycles_per_second: f32,
}

impl ConsciousnessSnapshot {
    /// Compute overall consciousness level from components.
    ///
    /// Modality-agnostic: accepts inputs from ANY source (text, sensors, body).
    /// Used by both `cycle(text)` and `cycle_with_hv(hv)` paths.
    pub fn compute_consciousness_level(
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
            ActionHint::Inhibit => actions.push("Inhibit current action"),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a snapshot with sensible baseline values for testing.
    /// All fields start at "healthy defaults" — individual tests override as needed.
    fn baseline_snapshot() -> ConsciousnessSnapshot {
        ConsciousnessSnapshot {
            cycle: 100,
            consciousness_level: 0.6,
            pattern: ConsciousnessPattern::Focused,
            pattern_confidence: 0.8,
            prediction_error: 0.1,
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
            temporal_coherence: 0.6,
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
            spectral_mip_adapted: false,
            spectral_mip_active_dim_count: None,
            harmonies_alignment: 0.5,
            empathic_compassion: 0.5,
            sigma: None,
            integrity_critical: false,
            network_critical: false,
            avg_cycle_time_us: 100.0,
            cycles_per_second: 50.0,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // compute_consciousness_level
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn consciousness_level_all_zero() {
        let level = ConsciousnessSnapshot::compute_consciousness_level(0.0, 0.0, 0.0, 0.0);
        assert_eq!(level, 0.0);
    }

    #[test]
    fn consciousness_level_all_one() {
        let level = ConsciousnessSnapshot::compute_consciousness_level(1.0, 1.0, 1.0, 1.0);
        // 0.3 + 0.25 + 0.2 + 0.25 = 1.0
        assert!((level - 1.0).abs() < 1e-6);
    }

    #[test]
    fn consciousness_level_weighted_correctly() {
        // Only prediction_confidence = 1.0, rest = 0.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(1.0, 0.0, 0.0, 0.0);
        assert!((level - 0.3).abs() < 1e-6);

        // Only temporal_coherence = 1.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(0.0, 1.0, 0.0, 0.0);
        assert!((level - 0.25).abs() < 1e-6);

        // Only flow_intensity = 1.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(0.0, 0.0, 1.0, 0.0);
        assert!((level - 0.2).abs() < 1e-6);

        // Only pattern_confidence = 1.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(0.0, 0.0, 0.0, 1.0);
        assert!((level - 0.25).abs() < 1e-6);
    }

    #[test]
    fn consciousness_level_clamped_high() {
        // Inputs > 1.0 should still clamp result to 1.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(2.0, 2.0, 2.0, 2.0);
        assert_eq!(level, 1.0);
    }

    #[test]
    fn consciousness_level_clamped_low() {
        // Negative inputs should clamp result to 0.0
        let level = ConsciousnessSnapshot::compute_consciousness_level(-1.0, -1.0, -1.0, -1.0);
        assert_eq!(level, 0.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // status()
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn status_contains_pattern() {
        let snap = baseline_snapshot();
        let status = snap.status();
        assert!(status.contains("Focused"), "status: {}", status);
    }

    #[test]
    fn status_shows_flow_when_in_flow() {
        let mut snap = baseline_snapshot();
        snap.in_flow = true;
        let status = snap.status();
        assert!(status.contains("FLOW"), "status: {}", status);
    }

    #[test]
    fn status_shows_explore_when_exploring() {
        let mut snap = baseline_snapshot();
        snap.exploring = true;
        let status = snap.status();
        assert!(status.contains("EXPLORE"), "status: {}", status);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // is_optimal()
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn is_optimal_when_assessment_optimal() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Optimal;
        assert!(snap.is_optimal());
    }

    #[test]
    fn is_optimal_when_in_flow_with_high_confidence() {
        let mut snap = baseline_snapshot();
        snap.in_flow = true;
        snap.prediction_confidence = 0.8;
        assert!(snap.is_optimal());
    }

    #[test]
    fn not_optimal_in_flow_with_low_confidence() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Learning;
        snap.in_flow = true;
        snap.prediction_confidence = 0.4;
        assert!(!snap.is_optimal());
    }

    #[test]
    fn not_optimal_baseline() {
        let snap = baseline_snapshot();
        assert!(!snap.is_optimal());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // needs_attention()
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn needs_attention_when_struggling() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Struggling;
        assert!(snap.needs_attention());
    }

    #[test]
    fn needs_attention_when_stagnating() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Stagnating;
        assert!(snap.needs_attention());
    }

    #[test]
    fn needs_attention_when_needs_calibration() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::NeedsCalibration;
        assert!(snap.needs_attention());
    }

    #[test]
    fn no_attention_needed_when_learning() {
        let snap = baseline_snapshot();
        assert!(!snap.needs_attention());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // dominant_concern()
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn dominant_concern_struggling() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Struggling;
        assert_eq!(
            snap.dominant_concern(),
            Some("High prediction error - system is struggling")
        );
    }

    #[test]
    fn dominant_concern_stagnating() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Stagnating;
        assert_eq!(
            snap.dominant_concern(),
            Some("Low error but no exploration - system is stagnating")
        );
    }

    #[test]
    fn dominant_concern_high_boredom() {
        let mut snap = baseline_snapshot();
        snap.boredom = 0.8;
        snap.exploring = false;
        assert_eq!(
            snap.dominant_concern(),
            Some("High boredom - needs novel input")
        );
    }

    #[test]
    fn dominant_concern_low_confidence() {
        let mut snap = baseline_snapshot();
        snap.prediction_confidence = 0.2;
        assert_eq!(
            snap.dominant_concern(),
            Some("Low confidence - predictions unreliable")
        );
    }

    #[test]
    fn dominant_concern_needs_calibration() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::NeedsCalibration;
        assert_eq!(
            snap.dominant_concern(),
            Some("Many adjustments made - consider manual review")
        );
    }

    #[test]
    fn dominant_concern_none_when_healthy() {
        let snap = baseline_snapshot();
        assert!(snap.dominant_concern().is_none());
    }

    #[test]
    fn dominant_concern_priority_struggling_over_boredom() {
        let mut snap = baseline_snapshot();
        snap.self_assessment = SelfAssessment::Struggling;
        snap.boredom = 0.9; // also high boredom, but Struggling is checked first
        assert!(snap.dominant_concern().unwrap().contains("struggling"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // recommended_actions()
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn recommended_actions_continue_adds_nothing() {
        let snap = baseline_snapshot();
        // action_hint=Continue, boredom low, predictions_trustworthy, not in flow
        let actions = snap.recommended_actions();
        assert!(
            actions.is_empty(),
            "expected empty actions, got: {:?}",
            actions
        );
    }

    #[test]
    fn recommended_actions_slow_down() {
        let mut snap = baseline_snapshot();
        snap.action_hint = ActionHint::SlowDown;
        let actions = snap.recommended_actions();
        assert!(actions.contains(&"Reduce input rate"));
    }

    #[test]
    fn recommended_actions_speed_up() {
        let mut snap = baseline_snapshot();
        snap.action_hint = ActionHint::SpeedUp;
        let actions = snap.recommended_actions();
        assert!(actions.contains(&"Can increase input rate"));
    }

    #[test]
    fn recommended_actions_boredom_content_variation() {
        let mut snap = baseline_snapshot();
        snap.boredom = 0.6;
        snap.exploring = false;
        let actions = snap.recommended_actions();
        assert!(actions.contains(&"Consider varying input content"));
    }

    #[test]
    fn recommended_actions_untrustworthy_predictions() {
        let mut snap = baseline_snapshot();
        snap.predictions_trustworthy = false;
        let actions = snap.recommended_actions();
        assert!(actions.contains(&"Predictions currently unreliable"));
    }

    #[test]
    fn recommended_actions_in_flow() {
        let mut snap = baseline_snapshot();
        snap.in_flow = true;
        let actions = snap.recommended_actions();
        assert!(actions.contains(&"In flow state - optimal for learning"));
    }

    #[test]
    fn recommended_actions_multiple_conditions() {
        let mut snap = baseline_snapshot();
        snap.action_hint = ActionHint::Explore;
        snap.boredom = 0.8;
        snap.predictions_trustworthy = false;
        snap.in_flow = true;
        let actions = snap.recommended_actions();
        assert!(
            actions.len() >= 4,
            "expected >= 4 actions, got: {:?}",
            actions
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Serialization
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn snapshot_serialization_roundtrip() {
        let snap = baseline_snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let deserialized: ConsciousnessSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.cycle, snap.cycle);
        assert_eq!(deserialized.pattern, snap.pattern);
        assert_eq!(deserialized.self_assessment, snap.self_assessment);
        assert_eq!(deserialized.in_flow, snap.in_flow);
    }
}
