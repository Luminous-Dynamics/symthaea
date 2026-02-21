//! Cognitive loop statistics.
//!
//! The `LoopStats` struct accumulates metrics across cognitive cycles.
//! Extracted from `mod.rs` to reduce file size.

use serde::{Deserialize, Serialize};

/// Statistics for the cognitive loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoopStats {
    /// Total cycles completed
    pub total_cycles: usize,

    /// Average prediction error (EMA)
    pub avg_prediction_error: f32,

    /// Learning cycles (error > threshold)
    pub learning_cycles: usize,

    /// Average training loss (EMA)
    pub avg_training_loss: f32,

    /// Attention variance (emergence metric)
    pub attention_variance: f32,

    /// Number of primitives with diverged attention
    pub diverged_primitives: usize,

    /// Buffer utilization (0-1)
    pub buffer_utilization: f32,

    /// Average cycle time (microseconds)
    pub avg_cycle_time_us: f32,

    /// Cycles per second
    pub cycles_per_second: f32,

    /// Prediction error trend (negative = improving)
    pub error_trend: f32,

    /// LTC consciousness level
    pub ltc_consciousness: f32,

    /// Temporal coherence (from CfC tau values)
    pub temporal_coherence: f32,

    /// Coherence-modulated learning rate
    pub effective_learning_rate: f32,

    /// Phi contribution from temporal coherence
    pub coherence_phi_contribution: f32,

    /// Voice articulation quality (0.0 to 1.0)
    pub voice_articulation_quality: f32,

    /// Voice rate stability (0.0 to 1.0)
    pub voice_rate_stability: f32,

    /// Phi adjustment from voice feedback
    pub voice_phi_adjustment: f32,

    /// Combined phi (coherence + voice contributions)
    pub combined_phi_contribution: f32,

    /// Current consciousness pattern (from temporal signatures)
    pub consciousness_pattern: String,

    /// Confidence in consciousness pattern classification
    pub pattern_confidence: f32,

    /// Tau trajectory mean
    pub tau_mean: f32,

    /// Tau trajectory trend
    pub tau_trend: f32,

    /// Current adaptive behavior state
    pub adaptive_confidence: f32,

    /// Current action hint
    pub action_hint: String,

    /// Whether learning is paused due to state transition
    pub learning_paused: bool,

    /// Adaptive learning rate (after all modulations)
    pub adaptive_learning_rate: f32,

    /// Adaptive speech rate multiplier
    pub adaptive_speech_rate: f32,

    /// Prediction confidence (decays during uncertain states)
    /// High when predictions are accurate and state is stable
    pub prediction_confidence: f32,

    /// Prediction confidence decay rate (higher = faster decay)
    pub confidence_decay_rate: f32,

    /// Whether currently in flow state
    pub in_flow: bool,

    /// Flow state intensity (0.0 to 1.0)
    pub flow_intensity: f32,

    /// Consecutive cycles in flow-compatible state
    pub flow_streak: u32,

    /// Learning boost from flow state
    pub flow_learning_boost: f32,

    /// Emotional valence from content (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// Emotional arousal (0.0 to 1.0)
    pub emotional_arousal: f32,

    /// Suggested pattern from emotion contagion
    pub emotion_nudge_pattern: String,

    /// Strength of emotion influence
    pub emotion_nudge_strength: f32,

    /// Boredom level from curiosity drive (0.0 to 1.0)
    pub boredom: f32,

    /// Curiosity level (0.0 to 1.0)
    pub curiosity: f32,

    /// Exploration urge (0.0 to 1.0)
    pub exploration_urge: f32,

    /// Whether curiosity-triggered exploration is active
    pub curiosity_exploring: bool,

    /// Novelty bonus to learning rate
    pub novelty_bonus: f32,

    // ===== Self-Reflection Stats =====
    /// Current self-assessment
    pub self_assessment: String,

    /// Number of reflection cycles performed
    pub reflection_count: u64,

    /// Threshold adjustments made
    pub adjustments_made: u32,

    /// Learning effectiveness score (0.0 to 1.0)
    pub learning_effectiveness: f32,

    /// Cycles until next reflection
    pub next_reflection_in: u32,

    /// Adapted flow error threshold
    pub adapted_flow_threshold: f32,

    /// Adapted boredom threshold
    pub adapted_boredom_threshold: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // MEGA-UNIFIED ARCHITECTURE STATS
    // ═══════════════════════════════════════════════════════════════════════════
    /// Current cognitive depth (from Thalamic routing)
    pub cognitive_depth: String,

    /// Unified Φ from the unification engine
    pub unified_psi: f32,

    /// Unified emotional valence (VAD-based, from EmotionalBridge)
    pub unified_emotional_valence: f32,

    /// Unified emotional arousal (VAD-based)
    pub unified_emotional_arousal: f32,

    /// Unified emotional dominance (VAD-based)
    pub unified_emotional_dominance: f32,

    /// Discrete emotion from unified bridge
    pub unified_emotion: String,

    /// Emotional pattern detected (Stable/Escalating/Calming/Volatile)
    pub emotional_pattern: String,

    /// Thalamic routing: fraction using Reflex path
    pub thalamic_reflex_rate: f32,

    /// Thalamic routing: fraction using Cortical path
    pub thalamic_cortical_rate: f32,

    /// Thalamic routing: fraction using DeepThought path
    pub thalamic_deep_rate: f32,

    /// Active Inference: Modulation Index (prediction-outcome coupling)
    pub active_inference_modulation_index: f32,

    /// Active Inference: Coupling quality
    pub active_inference_coupling_quality: String,

    /// Active Inference: Average prediction error (from PAC)
    pub active_inference_avg_error: f32,

    /// Enhanced FEP: Learning signal (for downstream systems)
    pub fep_learning_signal: f32,

    /// Enhanced FEP: Attention shift amount
    pub attention_shift: f32,

    /// Enhanced FEP: Action-outcome coupling quality
    pub fep_action_outcome_coupling: f32,

    /// Closed Learning Loop: Current strategy
    pub current_strategy: String,

    /// Closed Learning Loop: Best strategy (from Q-values)
    pub best_strategy: String,

    /// Closed Learning Loop: Average reward
    pub average_reward: f32,

    /// Closed Learning Loop: Exploration rate
    pub exploration_rate: f32,

    /// Closed Learning Loop: Total interactions
    pub learning_loop_interactions: u64,

    /// Episodic Memory: Short-term count
    pub memory_short_term_count: usize,

    /// Episodic Memory: Long-term count
    pub memory_long_term_count: usize,

    /// Episodic Memory: Total encoded
    pub memory_total_encoded: u64,

    /// World Model: Average prediction error across levels
    pub world_model_avg_error: f32,

    /// Active goals count
    pub active_goals_count: usize,

    /// Training cycles that used BPTT
    pub bptt_steps: u64,

    /// Training cycles that fell back to SPSA
    pub spsa_fallback_steps: u64,

    // ═══════════════════════════════════════════════════════════════════════════
    // SEMANTIC MEMORY STATS (HDC-based similarity lookup for CfC context)
    // ═══════════════════════════════════════════════════════════════════════════
    /// Semantic Memory: Number of queries that found similar entries
    pub semantic_hits: u64,

    /// Semantic Memory: Number of queries that found no similar entries
    pub semantic_misses: u64,

    /// Semantic Memory: Current learning rate factor from semantic context
    pub semantic_lr_factor: f32,

    /// Semantic Memory: Average prediction error of retrieved similar entries
    pub semantic_avg_retrieved_error: f32,

    /// Semantic Memory: Total entries stored
    pub semantic_entries_stored: u64,

    // ═══════════════════════════════════════════════════════════════════════════
    // ONLINE LEARNING STATS (Inference-time adaptation)
    // ═══════════════════════════════════════════════════════════════════════════
    /// Online Learning: Total adaptation calls
    pub online_adaptation_calls: u64,

    /// Online Learning: Adaptations that modified weights
    pub online_adaptations_applied: u64,

    /// Online Learning: Adaptations skipped due to low error
    pub online_adaptations_skipped: u64,

    /// Online Learning: EMA of prediction errors during online learning
    pub online_ema_error: f32,

    /// Online Learning: Cumulative weight change from online adaptation
    pub online_cumulative_weight_change: f32,

    // ═══════════════════════════════════════════════════════════════════════════
    // MORAL ALGEBRA STATS (Compositional ethical reasoning)
    // ═══════════════════════════════════════════════════════════════════════════
    /// Moral: Total moral evaluations performed
    pub moral_evaluations: u64,

    /// Moral: Number of inputs with moral concerns detected
    pub moral_concerns_detected: u64,

    /// Moral: Current moral score of last evaluation (-1.0 to 1.0)
    pub moral_score: f32,

    /// Moral: Whether last input had consent violation
    pub consent_violation: bool,

    /// Moral: Count of deontological violations in last input
    pub deontological_violations: usize,

    /// Moral: Whether current action needs ethical review
    pub moral_review_needed: bool,

    // ═══════════════════════════════════════════════════════════════════════════
    // CACHED LATTICE PROPERTIES (computed once — lattice is immutable)
    // ═══════════════════════════════════════════════════════════════════════════
    /// Cached lattice height (0 = not yet computed)
    pub lattice_height_cached: usize,
    /// Cached lattice width (0 = not yet computed)
    pub lattice_width_cached: usize,
}
