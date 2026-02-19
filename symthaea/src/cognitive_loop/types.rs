//! Public types returned by the cognitive loop.
//!
//! Extracted from `mod.rs` to reduce file size while preserving all public APIs.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE CARRYOVER — state that crosses cycle boundaries
// ═══════════════════════════════════════════════════════════════════════════════

/// State carried over between consecutive cognitive cycles.
///
/// These fields represent the "memory" of the previous cycle that influences
/// the next cycle's processing. All fields are reset to defaults by
/// `CognitiveLoopService::reset()`.
#[derive(Debug, Clone)]
pub(crate) struct CycleCarryover {
    /// Predictive processing phi modulation (1.0 = neutral)
    pub(crate) predictive_phi_modulation: f64,
    /// Cross-modal Phi (fed back into confidence)
    pub(crate) cross_modal_phi: f64,
    /// MCTS plan action (action_idx, confidence) for next cycle
    pub(crate) mcts_plan: Option<(usize, f32)>,
    /// Body phi modulation (fed back into unified_phi)
    pub(crate) body_phi_modulation: f64,
    /// Body arousal (fed back into CfC tau modulation)
    pub(crate) body_arousal: f32,
    /// Embodied cognition phi modulation (fed back into unified_phi)
    pub(crate) embodied_phi_modulation: f64,
    /// Resonance frequency (fed back into delta_t modulation)
    pub(crate) resonance_frequency: f64,
    /// Quantum coherence level (fed back into exploration boost)
    pub(crate) quantum_coherence: f64,
    /// Urgency level (hysteresis — prevents jitter)
    pub(crate) urgency: CycleUrgency,
    /// Prediction confidence snapshot at cycle start (drift clamping)
    pub(crate) prediction_confidence: f32,
    /// Whether narrative-GWT vetoed the previous cycle (suppresses learning)
    pub(crate) narrative_veto_active: bool,
    /// Consecutive cycles with error below threshold (Cruise mode trigger)
    pub(crate) consecutive_low_error: u32,
    /// MCE consciousness-level LR boost (decays 10%/cycle between MCE firings)
    pub(crate) mce_lr_boost: f32,
    /// Adaptive learning threshold multiplier (1.0 = config value as-is)
    pub(crate) adaptive_threshold_scale: f32,
    /// Consecutive high-arousal cycles (Yerkes-Dodson trap detection)
    pub(crate) arousal_trap_counter: u32,
    /// Last MCE consciousness level for learning gating
    pub(crate) consciousness_level: f64,
    /// Subsystem LR modulation factor (accumulated post-training, consumed next cycle).
    /// Meta-cognition, predictive processing, predictive self, phenomenal binding,
    /// and thermodynamics each multiply this to influence the NEXT cycle's training LR.
    /// Default 1.0 (neutral).
    pub(crate) subsystem_lr_factor: f32,
}

impl Default for CycleCarryover {
    fn default() -> Self {
        Self {
            predictive_phi_modulation: 1.0,
            cross_modal_phi: 0.0,
            mcts_plan: None,
            body_phi_modulation: 1.0,
            body_arousal: 0.5,
            embodied_phi_modulation: 1.0,
            resonance_frequency: 0.0,
            quantum_coherence: 0.0,
            urgency: CycleUrgency::Normal,
            prediction_confidence: 0.5,
            narrative_veto_active: false,
            consecutive_low_error: 0,
            mce_lr_boost: 0.0,
            adaptive_threshold_scale: 1.0,
            arousal_trap_counter: 0,
            consciousness_level: 0.0,
            subsystem_lr_factor: 1.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE URGENCY — adaptive subsystem scheduling
// ═══════════════════════════════════════════════════════════════════════════════

/// Urgency level controlling how many subsystems run each cycle.
///
/// Instead of fixed "every Nth cycle" throttling, urgency adapts to the
/// system's current needs:
/// - **Critical**: High error or surprise — run everything for maximum adaptation
/// - **Normal**: Standard processing — run most subsystems
/// - **Cruise**: Low error, stable state — skip expensive subsystems to save compute
///
/// Subsystems decide per-urgency whether to run:
/// - Core pipeline (HDC→CfC→predict→learn): always runs
/// - Moral evaluation: Critical+Normal (skip in Cruise unless new input)
/// - Enhanced FEP: Critical always, Normal every 4th, Cruise every 8th
/// - Stability regime: Critical+Normal, Cruise every 4th
/// - Consciousness monitors (resonance, quantum, temporal): Normal+Critical only
/// - Master equation: Critical every 5th, Normal every 10th, Cruise every 20th
/// - Body awareness (virtual body, affective, embodied): Normal+Critical, Cruise every 2nd
/// - Self models (meta-cognition, narrative, predictive mind/self): C=1, N=2, Cr=4
/// - Workspace (attention schema, GWT, cross-modal, narrative-GWT): C=1, N=2, Cr=4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CycleUrgency {
    /// High prediction error or surprise — run all subsystems
    Critical,
    /// Standard processing
    #[default]
    Normal,
    /// Low error, stable state — minimal subsystem overhead
    Cruise,
}

impl CycleUrgency {
    /// Compute urgency from current cycle state.
    ///
    /// - `prediction_error`: current cycle's prediction error
    /// - `learning_threshold`: config threshold for "significant" error
    /// - `surprise_triggered`: whether the surprise bridge triggered this cycle
    /// - `consecutive_low_error`: how many consecutive cycles have had error < threshold
    pub fn from_state(
        prediction_error: f32,
        learning_threshold: f32,
        surprise_triggered: bool,
        consecutive_low_error: u32,
    ) -> Self {
        if surprise_triggered || prediction_error > learning_threshold * 3.0 {
            CycleUrgency::Critical
        } else if prediction_error > learning_threshold || consecutive_low_error < 10 {
            CycleUrgency::Normal
        } else {
            CycleUrgency::Cruise
        }
    }

    /// Whether this urgency level should run a subsystem at the given cycle interval.
    /// Returns true if the subsystem should run this cycle.
    #[inline]
    pub fn should_run(
        &self,
        cycle: usize,
        critical_interval: usize,
        normal_interval: usize,
        cruise_interval: usize,
    ) -> bool {
        let interval = match self {
            CycleUrgency::Critical => critical_interval,
            CycleUrgency::Normal => normal_interval,
            CycleUrgency::Cruise => cruise_interval,
        };
        interval == 0 || cycle % interval == 0
    }

    /// Whether to run expensive consciousness monitors (resonance, quantum, temporal).
    #[inline]
    pub fn run_consciousness_monitors(&self) -> bool {
        matches!(self, CycleUrgency::Critical | CycleUrgency::Normal)
    }
}

/// Metadata about internal decision-making during a cycle.
///
/// Provides observability into which subsystems influenced the cycle's output,
/// enabling debugging of "why did the agent do that?" questions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleMetadata {
    /// Whether the surprise exploration bridge triggered exploration this cycle
    pub surprise_triggered: bool,

    /// Whether the prefrontal cortex vetoed or modified the response
    pub prefrontal_veto: bool,

    /// Confidence score from the reasoning engine (0.0 = unused/off, >0 = active)
    pub reasoning_confidence: f32,

    /// Description of the exploration action taken (if any)
    pub exploration_action: Option<String>,

    /// Whether the reasoning engine's tool gate blocked an action this cycle.
    /// When true, the system used a fallback strategy instead.
    pub reasoning_gate_blocked: bool,

    /// Fallback strategy selected when gating blocked an action (if any)
    pub reasoning_fallback: Option<String>,

    /// Best action from MCTS planning (Tier 1+), if planning ran
    pub reasoning_plan_action: Option<usize>,

    /// MCTS plan confidence (0.0 = no plan, >0 = plan confidence)
    pub reasoning_plan_confidence: f32,

    /// Human-readable reasoning narrative (Tier 2, best-effort)
    pub reasoning_narrative: Option<String>,

    /// Meta-cognitive self-model accuracy (0.0 = uncertain, 1.0 = perfect self-knowledge)
    pub meta_cognitive_accuracy: f32,

    /// Meta-cognitive recursion depth (0 = off, 1 = basic, 2+ = recursive self-modeling)
    pub meta_cognitive_depth: u8,

    /// Narrative self-model's integrated information (0.0 = off/no self, >0 = active self-Φ)
    pub narrative_self_phi: f64,

    /// Virtual body phi modulation (1.0 = neutral, >1 = body boosts consciousness)
    pub body_phi_modulation: f64,

    /// Virtual body affect valence (-1 to 1)
    pub body_valence: f32,

    /// Virtual body affect arousal (0 to 1)
    pub body_arousal: f32,

    /// Master Consciousness Equation level (0.0 to 1.0).
    /// Comprehensive consciousness metric combining Phi, broadcast, working memory,
    /// attention, recurrence, embodiment, knowledge, narrative, and social factors.
    /// Updated every 10th cycle; 0.0 when not yet computed.
    pub consciousness_level: f64,

    /// Predictive self-model safety score (1.0 = safe, 0.0 = unsafe).
    /// 0.0 when predictive self is not enabled.
    pub predictive_self_safety: f32,

    /// Attention schema focus intensity (0.0 to 1.0).
    /// 0.0 when attention schema is not enabled.
    pub attention_schema_focus: f32,

    /// Whether a GWT broadcast occurred this cycle.
    pub gwt_broadcast: bool,

    /// Consciousness resonance dominant frequency (Hz).
    /// 0.0 when resonance is not enabled or no history.
    pub resonance_frequency: f64,

    /// Quantum coherence level (0.0 to 1.0).
    /// 0.0 when quantum coherence is not enabled.
    pub quantum_coherence_level: f64,

    /// Temporal consciousness coherence (0.0 to 1.0).
    /// 0.0 when temporal consciousness is not enabled.
    pub temporal_coherence_score: f64,

    /// Whether temporal consciousness analysis detected a discontinuity.
    pub temporal_discontinuity: bool,

    /// Embodied cognition phi modulation (1.0 = neutral).
    /// 1.0 when embodied cognition is not enabled.
    pub embodied_phi_modulation: f64,

    /// Embodied cognition agency score (0.0 to 1.0).
    /// 0.0 when embodied cognition is not enabled.
    pub embodied_agency: f64,

    /// Whether the narrative-GWT integration vetoed this cycle's action.
    pub narrative_gwt_veto: bool,

    /// Self-Phi from the narrative-GWT integration (0.0 = off/not enabled).
    pub narrative_gwt_self_phi: f64,

    /// Unified Living Mind vitality (0.0 to 1.0).
    /// Measures overall "aliveness" of the system via life-mind continuity.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_vitality: f64,

    /// Unified Living Mind coherence (0.0 to 1.0).
    /// Measures integration quality of autopoietic, enactive, and predictive subsystems.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_coherence: f64,

    /// Cycle urgency level (Critical/Normal/Cruise).
    /// Determines how many subsystems ran this cycle.
    pub urgency: CycleUrgency,

    /// Number of insights gained from dream replay this cycle (0 = no dreaming).
    pub dream_insights: usize,

    /// Best Phi improvement discovered by dream counterfactuals (0.0 = no improvement).
    pub dream_phi_improvement: f32,

    /// Total accumulated wisdom entries from dreaming.
    pub dream_wisdom_count: usize,

    /// Predictive processing free energy (0.0 when off).
    pub predictive_free_energy: f64,

    /// Predictive processing phi modulation (1.0 when off — neutral).
    pub predictive_phi_modulation: f64,

    /// Cross-modal binding strength (0.0 when off).
    pub cross_modal_binding_strength: f32,

    /// Cross-modal integration Phi (0.0 when off).
    pub cross_modal_phi: f64,

    /// Affective bridge valence (-1 to 1, 0.0 when off).
    pub affective_valence: f32,

    /// Affective bridge arousal (0 to 1, 0.5 when off — neutral).
    pub affective_arousal: f32,

    /// Consciousness thermodynamic entropy (0.0 when off).
    pub thermodynamic_entropy: f64,

    /// Consciousness thermodynamic free energy (0.0 when off).
    pub thermodynamic_free_energy: f64,

    /// Phenomenal binding strength Ψ (0.0 when off).
    pub phenomenal_binding_strength: f64,

    /// Whether phenomenal binding detected fragmentation.
    pub phenomenal_fragmented: bool,

    /// Hierarchical total free energy (0.0 when off).
    pub hierarchical_total_free_energy: f64,

    /// Rolling average of Phi observations from phi_attention (0.0 when off).
    pub phi_attention_avg: f32,

    /// Phi estimate from primitive consciousness decomposition (0.0 when off).
    pub primitive_phi: f64,

    /// Negation polarity detected in input text (0.0 = no negation, >0.5 = negated).
    pub negation_polarity: f32,

    /// Moral judgment score for this cycle (-1.0 to 1.0). 0.0 when moral evaluation was skipped.
    pub moral_score: f32,

    /// Selected response strategy for this cycle (e.g., "Exploratory", "Supportive").
    pub selected_strategy: String,

    /// Actual effective learning rate used for training this cycle (after all modulations).
    /// 0.0 when no learning occurred.
    pub actual_effective_lr: f32,

    /// Cycle reward signal (internal + external blend, -1.0 to 1.0).
    pub cycle_reward: f32,

    /// FEP action index selected this cycle (0=exploit, 1=consolidate, 2=explore, 3=tighten).
    pub fep_action: usize,

    /// Per-module timing (microseconds). 0 = module disabled or not run this cycle.
    pub module_timings_us: ModuleTimings,
}

/// Compact subset of CycleMetadata with the most essential telemetry fields.
///
/// Use `CycleMetadata::compact()` to extract this from a full metadata struct.
/// Useful for lightweight logging, streaming telemetry, or consumers that only
/// need high-level consciousness state rather than per-subsystem detail.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleMetadataCompact {
    pub surprise_triggered: bool,
    pub prefrontal_veto: bool,
    pub reasoning_confidence: f32,
    pub consciousness_level: f64,
    pub gwt_broadcast: bool,
    pub urgency: CycleUrgency,
    pub body_phi_modulation: f64,
    pub meta_cognitive_accuracy: f32,
    pub narrative_self_phi: f64,
    pub affective_valence: f32,
    pub affective_arousal: f32,
    pub prediction_error_trend: f32,
}

impl CycleMetadata {
    /// Extract a compact subset of the most essential telemetry fields.
    pub fn compact(&self) -> CycleMetadataCompact {
        CycleMetadataCompact {
            surprise_triggered: self.surprise_triggered,
            prefrontal_veto: self.prefrontal_veto,
            reasoning_confidence: self.reasoning_confidence,
            consciousness_level: self.consciousness_level,
            gwt_broadcast: self.gwt_broadcast,
            urgency: self.urgency,
            body_phi_modulation: self.body_phi_modulation,
            meta_cognitive_accuracy: self.meta_cognitive_accuracy,
            narrative_self_phi: self.narrative_self_phi,
            affective_valence: self.affective_valence,
            affective_arousal: self.affective_arousal,
            prediction_error_trend: 0.0, // caller fills from CycleResult
        }
    }
}

/// Per-module execution timings in microseconds for overhead profiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleTimings {
    pub affective_bridge: u64,
    pub predictive_processing: u64,
    pub cross_modal_binding: u64,
    pub surprise_exploration: u64,
    pub prefrontal: u64,
    pub meta_cognition: u64,
    pub narrative_self: u64,
    pub gwt: u64,
    pub virtual_body: u64,
    pub embodied_cognition: u64,
    pub dream_replay: u64,
    pub moral_algebra: u64,
    pub consciousness_resonance: u64,
    pub temporal_consciousness: u64,
    pub attention_schema: u64,
    pub narrative_gwt: u64,
    pub consciousness_thermodynamics: u64,
    pub phenomenal_binding: u64,
    pub hierarchical_free_energy: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTESTATION RECORD — for governance bridge consumption
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of a Phi measurement from a cognitive cycle, ready for attestation.
///
/// The cognitive loop produces these after each cycle where Phi is computed.
/// The personal cluster (or symthaea-mycelix-bridge) can consume these records,
/// sign them with the agent's cryptographic key, and submit to governance as
/// authenticated `PhiAttestation` entries.
///
/// This is a lightweight struct with no external dependencies — the bridge crate
/// converts it to the Holochain-compatible `PhiAttestationData` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiAttestationRecord {
    /// Unified Phi value from this cycle (clamped to [0.0, 1.0])
    pub phi: f64,

    /// Cognitive cycle number that produced this Phi
    pub cycle_id: u64,

    /// Timestamp in microseconds since UNIX epoch
    pub captured_at_us: u64,

    /// Prediction error at time of measurement (context for Phi quality)
    pub prediction_error: f32,

    /// Urgency level during measurement (Critical/Normal/Cruise)
    pub urgency: CycleUrgency,
}

impl PhiAttestationRecord {
    /// Canonical message for signing: deterministic byte representation.
    /// The bridge crate signs this with the agent's Ed25519 key.
    ///
    /// Phi is formatted to 6 decimal places (`{:.6}`), matching the governance
    /// bridge's reconstruction format in `record_phi_attestation`. This precision
    /// (~0.000001) is sufficient for IIT Phi and ensures signature verification
    /// succeeds across Symthaea → bridge → governance without floating-point drift.
    pub fn sign_message(&self, agent_did: &str) -> Vec<u8> {
        format!(
            "symthaea-phi-attestation:v1:{}:{:.6}:{}:{}",
            agent_did, self.phi, self.cycle_id, self.captured_at_us,
        )
        .into_bytes()
    }
}

/// Result of a single cognitive cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// LTC output (interpretation of current state)
    pub output: Vec<f32>,

    /// Prediction error for this cycle
    pub prediction_error: f32,

    /// Peak attention value from encoder
    pub peak_attention: f32,

    /// Detected primitives in input
    pub detected_primitives: Vec<String>,

    /// Whether learning occurred this cycle
    pub learning_occurred: bool,

    /// Training loss (if learning occurred)
    pub training_loss: Option<f32>,

    /// Cycle timing (microseconds)
    pub cycle_time_us: u64,

    /// Internal decision-making metadata for observability
    pub metadata: CycleMetadata,

    /// Signed output for identity verification (when identity feature enabled)
    /// Contains Ed25519 signature over output hash and agent metadata
    #[cfg(feature = "identity")]
    pub signed_output: Option<crate::identity::SignedOutput>,

    /// Agent assurance level at time of processing (when identity feature enabled)
    #[cfg(feature = "identity")]
    pub assurance_level: crate::identity::AssuranceLevel,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL JUDGMENT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

/// Summary of moral evaluation for an action or input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralJudgmentSummary {
    /// The input text that was evaluated
    pub input: String,

    /// Overall moral verdict (Good, Bad, Neutral, ConsentViolation)
    pub verdict: String,

    /// Deontological verdict (Permissible, Impermissible, Obligatory, Supererogatory)
    pub deontological_verdict: String,

    /// List of detected moral violations (e.g., "dishonesty", "theft")
    pub violations: Vec<String>,

    /// List of moral satisfactions (e.g., "honesty", "beneficence")
    pub satisfactions: Vec<String>,

    /// Whether consent was violated
    pub consent_violation: bool,

    /// Moral score (-1.0 to 1.0, negative = bad, positive = good)
    pub moral_score: f32,

    /// Confidence in the moral judgment (0.0 to 1.0)
    pub confidence: f32,
}

impl Default for MoralJudgmentSummary {
    fn default() -> Self {
        Self {
            input: String::new(),
            verdict: "Neutral".to_string(),
            deontological_verdict: "Permissible".to_string(),
            violations: Vec::new(),
            satisfactions: Vec::new(),
            consent_violation: false,
            moral_score: 0.0,
            confidence: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive behavior parameters derived from consciousness state
///
/// These parameters allow the system to self-regulate based on its
/// current consciousness pattern, creating a truly self-aware loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBehavior {
    /// Learning rate multiplier (0.1 to 2.0)
    /// Higher when focused/confident, lower when uncertain
    pub learning_rate_multiplier: f32,

    /// Speech rate multiplier (0.7 to 1.3)
    /// Slower when contemplative/uncertain, faster when excited/focused
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier (0.5 to 2.0)
    /// Longer pauses when contemplative, shorter when excited
    pub pause_multiplier: f32,

    /// Attention sensitivity (0.5 to 1.5)
    /// Higher sensitivity when exploratory, lower when focused
    pub attention_sensitivity: f32,

    /// Exploration factor (0.0 to 1.0)
    /// Higher when exploratory/uncertain, lower when focused
    pub exploration_factor: f32,

    /// Confidence level (0.0 to 1.0)
    /// Derived from pattern + coherence + voice quality
    pub confidence: f32,

    /// Should pause learning (e.g., during transitions)
    pub pause_learning: bool,

    /// Recommended action hint
    pub action_hint: ActionHint,
}

/// Recommended action based on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionHint {
    /// Continue normally
    Continue,
    /// Slow down, be more deliberate
    SlowDown,
    /// Speed up, more confident
    SpeedUp,
    /// Pause and stabilize
    Stabilize,
    /// Explore alternatives
    Explore,
    /// Seek clarification or more input
    SeekInput,
}

impl Default for AdaptiveBehavior {
    fn default() -> Self {
        Self {
            learning_rate_multiplier: 1.0,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            attention_sensitivity: 1.0,
            exploration_factor: 0.3,
            confidence: 0.5,
            pause_learning: false,
            action_hint: ActionHint::Continue,
        }
    }
}

impl AdaptiveBehavior {
    /// Compute adaptive behavior from consciousness pattern and metrics
    pub fn from_consciousness_state(
        pattern: crate::dynamics::temporal_signatures::ConsciousnessPattern,
        pattern_confidence: f32,
        coherence: f32,
        voice_confidence: f32,
    ) -> Self {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;

        // Base confidence from all sources
        let confidence =
            (pattern_confidence * 0.4 + coherence * 0.3 + voice_confidence * 0.3).clamp(0.0, 1.0);

        match pattern {
            ConsciousnessPattern::Focused => Self {
                learning_rate_multiplier: 1.3 + confidence * 0.4, // 1.3 to 1.7
                speech_rate_multiplier: 1.05 + confidence * 0.15, // 1.05 to 1.2
                pause_multiplier: 0.7,
                attention_sensitivity: 0.7, // Less distracted
                exploration_factor: 0.1,    // Stay on track
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SpeedUp,
            },

            ConsciousnessPattern::Contemplative => Self {
                learning_rate_multiplier: 0.8,
                speech_rate_multiplier: 0.85,
                pause_multiplier: 1.5, // Longer pauses for reflection
                attention_sensitivity: 1.0,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SlowDown,
            },

            ConsciousnessPattern::Excited => Self {
                learning_rate_multiplier: 1.1,
                speech_rate_multiplier: 1.15,
                pause_multiplier: 0.6,      // Quick transitions
                attention_sensitivity: 1.3, // More reactive
                exploration_factor: 0.4,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Exploratory => Self {
                learning_rate_multiplier: 1.0,
                speech_rate_multiplier: 0.95,
                pause_multiplier: 1.0,
                attention_sensitivity: 1.4, // High sensitivity to new info
                exploration_factor: 0.7,    // Actively explore
                confidence: confidence * 0.8,
                pause_learning: false,
                action_hint: ActionHint::Explore,
            },

            ConsciousnessPattern::Resting => Self {
                learning_rate_multiplier: 0.6,
                speech_rate_multiplier: 0.9,
                pause_multiplier: 1.2,
                attention_sensitivity: 0.8,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Transitioning => Self {
                learning_rate_multiplier: 0.3, // Minimal learning during transition
                speech_rate_multiplier: 0.8,
                pause_multiplier: 1.8, // Pause to stabilize
                attention_sensitivity: 1.0,
                exploration_factor: 0.3,
                confidence: confidence * 0.5,
                pause_learning: true, // Pause learning
                action_hint: ActionHint::Stabilize,
            },

            ConsciousnessPattern::Uncertain => Self {
                learning_rate_multiplier: 0.4, // Careful learning
                speech_rate_multiplier: 0.75,  // Slow down
                pause_multiplier: 2.0,         // Long pauses
                attention_sensitivity: 1.2,
                exploration_factor: 0.5,
                confidence: confidence * 0.3,
                pause_learning: false,
                action_hint: ActionHint::SeekInput,
            },
        }
    }

    /// Get effective learning rate with all modulations
    pub fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        if self.pause_learning {
            0.0
        } else {
            base_rate * self.learning_rate_multiplier
        }
    }

    /// Check if the system should seek more input/clarification
    pub fn should_seek_input(&self) -> bool {
        self.action_hint == ActionHint::SeekInput || self.confidence < 0.3
    }

    /// Check if the system is in a confident state
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.6 && !self.pause_learning
    }

    /// Get a human-readable description of current state
    pub fn description(&self) -> &'static str {
        match self.action_hint {
            ActionHint::Continue => "Operating normally",
            ActionHint::SlowDown => "Deliberating carefully",
            ActionHint::SpeedUp => "Confident and focused",
            ActionHint::Stabilize => "Stabilizing state",
            ActionHint::Explore => "Exploring possibilities",
            ActionHint::SeekInput => "Seeking clarification",
        }
    }
}
